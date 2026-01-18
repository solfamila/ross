/**
 * @file image_loader.cpp
 * @brief Static image loading implementation using Windows Imaging Component
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincodec.h>
#include <cstdlib>
#include <cstdint>

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

static const char* g_stbi_failure_reason = nullptr;

static void set_failure_reason(const char* reason) {
    g_stbi_failure_reason = reason;
}

extern "C" {

const char* stbi_failure_reason() {
    return g_stbi_failure_reason;
}

void stbi_image_free(void* data) {
    if (data) {
        free(data);
    }
}

unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels_in_file, int desired_channels) {
    if (!filename || !x || !y || !channels_in_file) {
        set_failure_reason("Invalid parameters");
        return nullptr;
    }

    // Convert filename to wide string
    int wlen = MultiByteToWideChar(CP_UTF8, 0, filename, -1, nullptr, 0);
    if (wlen <= 0) {
        set_failure_reason("Failed to convert filename");
        return nullptr;
    }

    wchar_t* wfilename = static_cast<wchar_t*>(malloc(wlen * sizeof(wchar_t)));
    if (!wfilename) {
        set_failure_reason("Memory allocation failed");
        return nullptr;
    }
    MultiByteToWideChar(CP_UTF8, 0, filename, -1, wfilename, wlen);

    // Initialize COM
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool comInitialized = SUCCEEDED(hr) || hr == S_FALSE || hr == RPC_E_CHANGED_MODE;

    unsigned char* result = nullptr;
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;

    do {
        // Create WIC factory
        hr = CoCreateInstance(
            CLSID_WICImagingFactory,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&factory)
        );
        if (FAILED(hr)) {
            set_failure_reason("Failed to create WIC factory");
            break;
        }

        // Create decoder from file
        hr = factory->CreateDecoderFromFilename(
            wfilename,
            nullptr,
            GENERIC_READ,
            WICDecodeMetadataCacheOnDemand,
            &decoder
        );
        if (FAILED(hr)) {
            set_failure_reason("Failed to decode image file");
            break;
        }

        // Get first frame
        hr = decoder->GetFrame(0, &frame);
        if (FAILED(hr)) {
            set_failure_reason("Failed to get image frame");
            break;
        }

        // Get dimensions
        UINT width = 0, height = 0;
        hr = frame->GetSize(&width, &height);
        if (FAILED(hr)) {
            set_failure_reason("Failed to get image size");
            break;
        }

        *x = static_cast<int>(width);
        *y = static_cast<int>(height);
        *channels_in_file = 4;  // We always convert to BGRA

        // Create format converter
        hr = factory->CreateFormatConverter(&converter);
        if (FAILED(hr)) {
            set_failure_reason("Failed to create format converter");
            break;
        }

        // Convert to 32bpp BGRA
        hr = converter->Initialize(
            frame,
            GUID_WICPixelFormat32bppBGRA,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0,
            WICBitmapPaletteTypeCustom
        );
        if (FAILED(hr)) {
            set_failure_reason("Failed to initialize format converter");
            break;
        }

        // Allocate output buffer
        UINT stride = width * 4;
        UINT bufSize = stride * height;
        result = static_cast<unsigned char*>(malloc(bufSize));
        if (!result) {
            set_failure_reason("Memory allocation failed for pixels");
            break;
        }

        // Copy pixels
        hr = converter->CopyPixels(nullptr, stride, bufSize, result);
        if (FAILED(hr)) {
            free(result);
            result = nullptr;
            set_failure_reason("Failed to copy pixels");
            break;
        }

    } while (false);

    // Cleanup
    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    free(wfilename);

    return result;
}

} // extern "C"

