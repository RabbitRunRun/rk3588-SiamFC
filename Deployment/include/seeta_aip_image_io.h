//
// Created by seeta on 2020/10/30.
//

#ifndef SEETA_AIP_SEETA_AIP_IMAGE_IO_H
#define SEETA_AIP_SEETA_AIP_IMAGE_IO_H

#include "seeta_aip_struct.h"
#include "seeta_aip_image.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image.h"
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image_write.h"

#include <cstring>
#include <cstdlib>
#include <iostream>

namespace seeta {
    namespace aip {
        static seeta::aip::ImageData imread(const std::string &filename) {
            int iw, ih, n;
            iw = ih = n = 0;
            unsigned char *data = stbi_load(filename.c_str(), &iw, &ih, &n, 3);
            if (!data) return seeta::aip::ImageData();
            seeta::aip::ImageData result(SEETA_AIP_FORMAT_U8RGB, iw, ih, n, data);
            stbi_image_free(data);
            return result;
        }

        static bool imwrite(const std::string &filename, const SeetaAIPImageData &image) {
            auto image_t = image;
            seeta::aip::ImageData tmp;
            if (image_t.format != SEETA_AIP_FORMAT_U8RGB && image_t.format != SEETA_AIP_FORMAT_U8RGBA) {
                tmp = convert(1, SEETA_AIP_FORMAT_U8RGB, image_t);
                image_t = tmp;
            }
            auto dot_pos = filename.rfind('.');
            std::string ext = dot_pos == std::string::npos ? "" : filename.substr(dot_pos + 1);
            for (auto &ch : ext) ch = std::tolower(ch);

            if (ext.empty()) {
                std::cerr << "Unknown image type, image not of any known type, or corrupt" << std::endl;
                return false;
            }

            if (ext == "jpeg" || ext == "jpg") {
                return stbi_write_jpg(filename.c_str(),
                                      image_t.width, image_t.height, image_t.channels, image_t.data,
                                      90);
            } else if (ext == "png") {
                return stbi_write_png(filename.c_str(),
                                      image_t.width, image_t.height, image_t.channels, image_t.data,
                                      0);
            } else if (ext == "bmp") {
                return stbi_write_bmp(filename.c_str(),
                                      image_t.width, image_t.height, image_t.channels,
                                      image_t.data);;
            } else if (ext == "tga") {
                return stbi_write_tga(filename.c_str(),
                                      image_t.width, image_t.height, image_t.channels, image_t.data);
            } else if (ext == "hdr") {
                return stbi_write_hdr(filename.c_str(),
                                      image_t.width, image_t.height, image_t.channels,
                                      (float *) image_t.data);;
            }

            std::cerr << "Unknown image type \'" << ext << "\', image not of any known type, or corrupt" << std::endl;
            return false;
        }
    }
}

#endif //SEETA_AIP_SEETA_AIP_IMAGE_IO_H
