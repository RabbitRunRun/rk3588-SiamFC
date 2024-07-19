//
// Created by SeetaTech on 2020/4/27.
//

#ifndef _INC_SEETA_AIP_OPENCV_H
#define _INC_SEETA_AIP_OPENCV_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "seeta_aip.h"
#include "seeta_aip_struct.h"

class SeetaAIPImageData;

namespace seeta {
    namespace aip {
        namespace opencv {
            class ImageData : public Wrapper< SeetaAIPImageData > {
            public:
                using self = ImageData;

                ImageData(uint32_t width,
                          uint32_t height,
                          uint8_t *data = nullptr) {
                    this->m_raw.format = SEETA_AIP_FORMAT_U8BGR;
                    this->m_raw.number = 1;
                    this->m_raw.width = width;
                    this->m_raw.height = height;
                    this->m_raw.channels = 3;
                    this->m_mat = ::cv::Mat(height, width, CV_8UC3, data);
                }

                ImageData(const ::cv::Mat &mat) {
                    if (mat.type() != CV_8UC3) throw Exception("Only support BGR image mat.");
                    this->m_raw.format = SEETA_AIP_FORMAT_U8BGR;
                    this->m_raw.number = 1;
                    this->m_raw.width = mat.cols;
                    this->m_raw.height = mat.rows;
                    this->m_raw.channels = mat.channels();
                    this->m_mat = mat.clone();
                }

                ImageData(::cv::Mat &&mat) {
                    if (mat.type() != CV_8UC3) throw Exception("Only support BGR image mat.");
                    this->m_raw.format = SEETA_AIP_FORMAT_U8BGR;
                    this->m_raw.number = 1;
                    this->m_raw.width = mat.cols;
                    this->m_raw.height = mat.rows;
                    this->m_raw.channels = mat.channels();
                    this->m_mat = std::forward<::cv::Mat>(mat);
                }

                explicit operator ::cv::Mat() const { return m_mat; }

                ::cv::Mat cvMat() const { return m_mat; }

                SEETA_AIP_VALUE_TYPE type() const { return SEETA_AIP_VALUE_BYTE; }

                _SEETA_AIP_WRAPPER_DECLARE_GETTER(format, SEETA_AIP_IMAGE_FORMAT)

                _SEETA_AIP_WRAPPER_DECLARE_GETTER(number, uint32_t)

                _SEETA_AIP_WRAPPER_DECLARE_GETTER(height, uint32_t)

                _SEETA_AIP_WRAPPER_DECLARE_GETTER(width, uint32_t)

                _SEETA_AIP_WRAPPER_DECLARE_GETTER(channels, uint32_t)

                uint32_t element_width() const {
                    switch (type()) {
                        default:
                            return 0;
                        case SEETA_AIP_VALUE_BYTE:
                            return 1;
                        case SEETA_AIP_VALUE_FLOAT32:
                            return 4;
                        case SEETA_AIP_VALUE_INT32:
                            return 4;
                        case SEETA_AIP_VALUE_FLOAT64:
                            return 8;
                    }
                }

                uint32_t bytes() const {
                    return number() * height() * width() * channels() * element_width();
                }

                uint8_t *data() { return m_mat.data; }

                const uint8_t *data() const { return m_mat.data; }

                uint8_t &data(size_t i) { return this->data()[i]; }

                const uint8_t &data(size_t i) const { return this->data()[i]; }

                uint8_t &data(int i) { return this->data()[i]; }

                const uint8_t &data(int i) const { return this->data()[i]; }

                std::vector<uint32_t> dims() const { return {number(), height(), width(), channels()}; }

                void exporter() override {
                    m_raw.data = m_mat.data;
                }

                void importer() override {
                    if (m_raw.format != SEETA_AIP_FORMAT_U8BGR) throw Exception("Only support BGR image mat.");
                    if (m_raw.number != 1) throw Exception("Only support number=1");
                    m_mat = ::cv::Mat(m_raw.height, m_raw.width, CV_8UC3, m_raw.data);
                }

            private:
                ::cv::Mat m_mat;
            };
        }
    }
}

#endif //_INC_SEETA_AIP_OPENCV_H
