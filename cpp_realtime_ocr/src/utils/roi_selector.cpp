#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "utils/roi_selector.h"

#include <Windows.h>
#include <windowsx.h>

#include <cstdint>

namespace trading_monitor {

namespace {

struct SelectorState {
    bool selecting = false;
    bool done = false;
    bool cancelled = false;

    POINT start{};
    POINT current{};
    RECT selected{};

    int width = 0;
    int height = 0;
};

static RECT normalizeRect(POINT a, POINT b) {
    RECT r{};
    r.left = std::min(a.x, b.x);
    r.top = std::min(a.y, b.y);
    r.right = std::max(a.x, b.x);
    r.bottom = std::max(a.y, b.y);
    return r;
}

static RECT clampRect(RECT r, int w, int h) {
    auto clampi = [](int v, int lo, int hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    };

    r.left = static_cast<LONG>(clampi(static_cast<int>(r.left), 0, w));
    r.right = static_cast<LONG>(clampi(static_cast<int>(r.right), 0, w));
    r.top = static_cast<LONG>(clampi(static_cast<int>(r.top), 0, h));
    r.bottom = static_cast<LONG>(clampi(static_cast<int>(r.bottom), 0, h));
    if (r.right < r.left) std::swap(r.right, r.left);
    if (r.bottom < r.top) std::swap(r.bottom, r.top);
    return r;
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<SelectorState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

    switch (msg) {
    case WM_CREATE: {
        auto* cs = reinterpret_cast<CREATESTRUCT*>(lParam);
        state = reinterpret_cast<SelectorState*>(cs->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
        SetFocus(hwnd);
        return 0;
    }
    case WM_SYSKEYDOWN:
    case WM_KEYDOWN: {
        if (!state) break;
        if (wParam == VK_ESCAPE) {
            state->cancelled = true;
            PostQuitMessage(0);
            return 0;
        }
        return 0;
    }
    case WM_RBUTTONDOWN: {
        if (!state) break;
        state->cancelled = true;
        PostQuitMessage(0);
        return 0;
    }
    case WM_LBUTTONDOWN: {
        if (!state) break;
        state->selecting = true;
        state->start = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
        state->current = state->start;
        SetCapture(hwnd);
        InvalidateRect(hwnd, nullptr, TRUE);
        return 0;
    }
    case WM_MOUSEMOVE: {
        if (!state) break;
        if (state->selecting) {
            state->current = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
            InvalidateRect(hwnd, nullptr, TRUE);
        }
        return 0;
    }
    case WM_LBUTTONUP: {
        if (!state) break;
        if (state->selecting) {
            ReleaseCapture();
            state->selecting = false;
            state->current = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
            RECT r = normalizeRect(state->start, state->current);
            r = clampRect(r, state->width, state->height);

            const int rw = r.right - r.left;
            const int rh = r.bottom - r.top;
            if (rw >= 2 && rh >= 2) {
                state->selected = r;
                state->done = true;
                PostQuitMessage(0);
            } else {
                // Too small; keep selecting.
                InvalidateRect(hwnd, nullptr, TRUE);
            }
        }
        return 0;
    }
    case WM_PAINT: {
        if (!state) break;

        PAINTSTRUCT ps{};
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT client{};
        GetClientRect(hwnd, &client);

        // Semi-transparent overlay (whole window alpha is set via WS_EX_LAYERED).
        HBRUSH bg = CreateSolidBrush(RGB(0, 0, 0));
        FillRect(hdc, &client, bg);
        DeleteObject(bg);

        // Instructions text
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(255, 255, 255));
        const wchar_t* text = L"Drag to select ROI. ESC to cancel.";
        RECT textRc{ 10, 10, client.right - 10, 40 };
        DrawTextW(hdc, text, -1, &textRc, DT_LEFT | DT_TOP | DT_SINGLELINE);

        // Draw current selection rectangle
        if (state->selecting || state->done) {
            RECT r = state->selecting
                ? clampRect(normalizeRect(state->start, state->current), state->width, state->height)
                : state->selected;

            HPEN pen = CreatePen(PS_SOLID, 2, RGB(0, 200, 255));
            HGDIOBJ oldPen = SelectObject(hdc, pen);
            HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));

            Rectangle(hdc, r.left, r.top, r.right, r.bottom);

            SelectObject(hdc, oldBrush);
            SelectObject(hdc, oldPen);
            DeleteObject(pen);

            // Show coords
            wchar_t buf[256];
            wsprintfW(buf, L"x=%d y=%d w=%d h=%d", r.left, r.top, r.right - r.left, r.bottom - r.top);
            RECT coordRc{ 10, 45, client.right - 10, 75 };
            DrawTextW(hdc, buf, -1, &coordRc, DT_LEFT | DT_TOP | DT_SINGLELINE);
        }

        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        break;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

} // namespace

bool selectROIInteractive(void* hwndVoid, ROI& outRoi, std::string& errorMessage) {
    errorMessage.clear();

    HWND hwndTarget = reinterpret_cast<HWND>(hwndVoid);
    if (!hwndTarget || !IsWindow(hwndTarget)) {
        errorMessage = "Invalid HWND";
        return false;
    }

    // Bring target window to foreground so user can see it.
    ShowWindow(hwndTarget, SW_RESTORE);
    SetForegroundWindow(hwndTarget);

    RECT rcClient{};
    if (!GetClientRect(hwndTarget, &rcClient)) {
        errorMessage = "GetClientRect failed";
        return false;
    }

    POINT tl{ 0, 0 };
    POINT br{ rcClient.right, rcClient.bottom };
    if (!ClientToScreen(hwndTarget, &tl) || !ClientToScreen(hwndTarget, &br)) {
        errorMessage = "ClientToScreen failed";
        return false;
    }

    int w = br.x - tl.x;
    int h = br.y - tl.y;
    if (w < 0) w = 0;
    if (h < 0) h = 0;
    if (w <= 0 || h <= 0) {
        errorMessage = "Target window client area is empty";
        return false;
    }

    SelectorState state;
    state.width = w;
    state.height = h;

    HINSTANCE hinst = GetModuleHandleW(nullptr);

    const wchar_t* className = L"TM_ROI_SELECTOR_OVERLAY";
    static std::atomic<bool> s_classRegistered{ false };
    if (!s_classRegistered.load()) {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = WndProc;
        wc.hInstance = hinst;
        wc.lpszClassName = className;
        wc.hCursor = LoadCursor(nullptr, IDC_CROSS);
        wc.hbrBackground = nullptr;

        if (!RegisterClassExW(&wc)) {
            DWORD err = GetLastError();
            // If already registered, ignore.
            if (err != ERROR_CLASS_ALREADY_EXISTS) {
                errorMessage = "RegisterClassExW failed";
                return false;
            }
        }
        s_classRegistered.store(true);
    }

    const DWORD exStyle = WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_LAYERED;
    const DWORD style = WS_POPUP;

    HWND overlay = CreateWindowExW(
        exStyle,
        className,
        L"ROI Selector",
        style,
        tl.x,
        tl.y,
        w,
        h,
        nullptr,
        nullptr,
        hinst,
        &state);

    if (!overlay) {
        errorMessage = "CreateWindowExW failed";
        return false;
    }

    // Make it translucent so the underlying window remains visible.
    SetLayeredWindowAttributes(overlay, 0, static_cast<BYTE>(45), LWA_ALPHA);

    ShowWindow(overlay, SW_SHOW);
    UpdateWindow(overlay);

    // Try hard to get keyboard input for ESC.
    SetForegroundWindow(overlay);
    SetActiveWindow(overlay);
    SetFocus(overlay);

    // Message loop until selection done/cancelled.
    MSG msg{};
    for (;;) {
        // ESC cancel even if overlay doesn't have focus.
        if ((GetAsyncKeyState(VK_ESCAPE) & 1) != 0) {
            state.cancelled = true;
            break;
        }

        BOOL got = GetMessage(&msg, nullptr, 0, 0);
        if (got <= 0) {
            break;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);

        if (state.done || state.cancelled) {
            break;
        }
    }

    DestroyWindow(overlay);

    if (state.cancelled || !state.done) {
        errorMessage = "Cancelled";
        return false;
    }

    const RECT r = state.selected;
    outRoi.x = r.left;
    outRoi.y = r.top;
    outRoi.w = r.right - r.left;
    outRoi.h = r.bottom - r.top;
    return true;
}

} // namespace trading_monitor
