/**
 * @file stb_image.h
 * @brief Simple image loading using Windows Imaging Component (WIC)
 *
 * Provides stbi_load compatible API for loading PNG/JPEG/BMP images on Windows.
 */
#pragma once

#ifndef STB_IMAGE_H_INCLUDED
#define STB_IMAGE_H_INCLUDED

#include <cstdint>
#include <cstdlib>

typedef unsigned char stbi_uc;

#ifdef __cplusplus
extern "C" {
#endif

// API declarations
stbi_uc* stbi_load(const char* filename, int* x, int* y, int* channels_in_file, int desired_channels);
void stbi_image_free(void* retval_from_stbi_load);
const char* stbi_failure_reason();

#ifdef __cplusplus
}
#endif

#endif // STB_IMAGE_H_INCLUDED

