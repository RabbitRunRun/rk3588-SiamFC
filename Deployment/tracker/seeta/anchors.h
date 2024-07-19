#ifndef _INC_ANCHORS_H
#define _INC_ANCHORS_H

#include <vector>
#include <cmath>

#include "cann.h"

class Anchors {
public:
    int stride = 0;
    std::vector<float> ratios;
    std::vector<float> scales;

    int anchor_num = 0;     // = scales.size() * ratios.size()
    ACLTensor anchors = {}; // .shape = [anchor_num, 4]

    Anchors(int stride, std::vector<float> ratios, std::vector<float> scales)
        : stride(stride), ratios(std::move(ratios)), scales(std::move(scales)) {
        this->generate_anchors();
    }

    void generate_anchors() {
        this->anchor_num = int(this->scales.size() * this->ratios.size());
        
        this->anchors = ACLTensor(std::vector<int64_t>({this->anchor_num, 4}));
        auto size = this->stride * this->stride;

        int i = 0;
        for (auto r : this->ratios) {
            auto ws = int(std::sqrt(size * 1.0f / r));
            auto hs = int(ws * r);

            for (auto s : this->scales) {
                auto w = ws * s;
                auto h = hs * s;
                this->anchors.at(i, 0) = -w * 0.5f;
                this->anchors.at(i, 1) = -h * 0.5f;
                this->anchors.at(i, 2) = w * 0.5f;
                this->anchors.at(i, 3) = h * 0.5f;
                ++i;
            }
        }
    }
};

#endif // _INC_ANCHORS_H

