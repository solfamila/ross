#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "utils/roi_overlay.h"

#include <Windows.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace trading_monitor {

namespace {

struct OverlayState {
    HWND target = nullptr;
    std::vector<ROI> rois;
    std::atomic<bool> running{ true };
    std::mutex mu;

    RECT lastClientScreen{};
};

static RECT getClientScreenRect(HWND hwnd) {
    RECT rcClient{};
    GetClientRect(hwnd, &rcClient);

    POINT tl{ 0, 0 };
    POINT br{ rcClient.right, rcClient.bottom };
    ClientToScreen(hwnd, &tl);
    ClientToScreen(hwnd, &br);

    RECT out{};
    out.left = tl.x;
    out.top = tl.y;
    out.right = br.x;
    out.bottom = br.y;
    return out;
}

static COLORREF pickColorForROI(const ROI& r) {
    if (r.name.find("_row") != std::string::npos) {
        return RGB(0, 200, 255);
    }
    if (r.name.find("table") != std::string::npos) {
        return RGB(255, 200, 0);
    }
    return RGB(0, 255, 0);
}

static LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<OverlayState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

    switch (msg) {
    case WM_CREATE: {
        auto* cs = reinterpret_cast<CREATESTRUCT*>(lParam);
        state = reinterpret_cast<OverlayState*>(cs->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
        // Black is the colorkey -> transparent.
        SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
        SetTimer(hwnd, 1, 250, nullptr);
        return 0;
    }
    case WM_TIMER: {
        if (!state) break;
        if (!state->target || !IsWindow(state->target)) {
            DestroyWindow(hwnd);
            return 0;
        }
        RECT rc = getClientScreenRect(state->target);
        if (rc.left != state->lastClientScreen.left || rc.top != state->lastClientScreen.top ||
            rc.right != state->lastClientScreen.right || rc.bottom != state->lastClientScreen.bottom) {
            state->lastClientScreen = rc;
            const int w = rc.right - rc.left;
            const int h = rc.bottom - rc.top;
            if (w > 0 && h > 0) {
                MoveWindow(hwnd, rc.left, rc.top, w, h, TRUE);
            }
        }
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }
    case WM_PAINT: {
        if (!state) break;

        PAINTSTRUCT ps{};
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT client{};
        GetClientRect(hwnd, &client);

        // Fill with colorkey (transparent).
        HBRUSH bg = CreateSolidBrush(RGB(0, 0, 0));
        FillRect(hdc, &client, bg);
        DeleteObject(bg);

        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(255, 255, 255));

        std::vector<ROI> roisCopy;
        {
            std::lock_guard<std::mutex> lock(state->mu);
            roisCopy = state->rois;
        }

        for (const auto& r : roisCopy) {
            RECT rr{};
            rr.left = r.x;
            rr.top = r.y;
            rr.right = r.x + r.w;
            rr.bottom = r.y + r.h;

            const COLORREF color = pickColorForROI(r);
            HPEN pen = CreatePen(PS_SOLID, 2, color);
            HGDIOBJ oldPen = SelectObject(hdc, pen);
            HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));

            Rectangle(hdc, rr.left, rr.top, rr.right, rr.bottom);

            SelectObject(hdc, oldBrush);
            SelectObject(hdc, oldPen);
            DeleteObject(pen);

            // Label
            std::wstring wname(r.name.begin(), r.name.end());
            RECT textRc{ rr.left + 4, rr.top + 2, rr.right, rr.bottom };
            DrawTextW(hdc, wname.c_str(), -1, &textRc, DT_LEFT | DT_TOP | DT_SINGLELINE);
        }

        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_DESTROY:
        KillTimer(hwnd, 1);
        PostQuitMessage(0);
        return 0;
    default:
        break;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

static DWORD WINAPI overlayThreadProc(LPVOID param) {
    auto* state = reinterpret_cast<OverlayState*>(param);
    if (!state) return 0;

    HINSTANCE hinst = GetModuleHandleW(nullptr);
    const wchar_t* className = L"TM_ROI_OVERLAY";

    static std::atomic<bool> s_classRegistered{ false };
    if (!s_classRegistered.load()) {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = OverlayWndProc;
        wc.hInstance = hinst;
        wc.lpszClassName = className;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = nullptr;

        if (!RegisterClassExW(&wc)) {
            DWORD err = GetLastError();
            if (err != ERROR_CLASS_ALREADY_EXISTS) {
                return 0;
            }
        }
        s_classRegistered.store(true);
    }

    RECT rc = getClientScreenRect(state->target);
    state->lastClientScreen = rc;
    const int w = rc.right - rc.left;
    const int h = rc.bottom - rc.top;
    if (w <= 0 || h <= 0) {
        return 0;
    }

    const DWORD exStyle = WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE;
    const DWORD style = WS_POPUP;

    HWND overlay = CreateWindowExW(
        exStyle,
        className,
        L"ROI Overlay",
        style,
        rc.left,
        rc.top,
        w,
        h,
        nullptr,
        nullptr,
        hinst,
        state);

    if (!overlay) {
        return 0;
    }

    ShowWindow(overlay, SW_SHOWNOACTIVATE);
    UpdateWindow(overlay);

    MSG msg{};
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

} // namespace

bool startROIOverlay(void* targetHwnd, const std::vector<ROI>& rois, ROIOverlayHandle& outHandle, std::string& errorMessage) {
    errorMessage.clear();
    outHandle = {};

    HWND hwndTarget = reinterpret_cast<HWND>(targetHwnd);
    if (!hwndTarget || !IsWindow(hwndTarget)) {
        errorMessage = "Invalid target window";
        return false;
    }

    auto* state = new (std::nothrow) OverlayState();
    if (!state) {
        errorMessage = "Out of memory";
        return false;
    }
    state->target = hwndTarget;
    state->rois = rois;

    DWORD threadId = 0;
    HANDLE th = CreateThread(nullptr, 0, overlayThreadProc, state, 0, &threadId);
    if (!th) {
        delete state;
        errorMessage = "CreateThread failed";
        return false;
    }

    // We don't have the overlay HWND immediately; thread will create it.
    outHandle.threadHandle = th;
    outHandle.threadId = static_cast<uint32_t>(threadId);
    outHandle.statePtr = reinterpret_cast<void*>(state);
    return true;
}

void stopROIOverlay(ROIOverlayHandle& handle) {
    auto* state = reinterpret_cast<OverlayState*>(handle.statePtr);
    HANDLE th = reinterpret_cast<HANDLE>(handle.threadHandle);

    if (handle.threadId != 0) {
        PostThreadMessage(static_cast<DWORD>(handle.threadId), WM_QUIT, 0, 0);
    }

    if (th) {
        // Give it a moment to exit naturally.
        WaitForSingleObject(th, 500);
        CloseHandle(th);
    }

    delete state;
    handle = {};
}

} // namespace trading_monitor
