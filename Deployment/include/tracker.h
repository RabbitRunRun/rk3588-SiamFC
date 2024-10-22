#ifndef _INC_TRACKER_H
#define _INC_TRACKER_H

#include <string>
#include <vector>
#include <cmath>
#include <array>
#include "cann.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "rknn_api.h"

struct ScoreRect {
    int   ret;
    float score;
    cv::Rect rect;
};


class Tracker {
public:
    std::vector<float> m_window;
    std::array<int,2> m_size;
    uint8_t m_channel_average[3] = {0,0,0};
    std::array<int, 2> m_center_pos;
    
    std::array<int, 2> m_template_size;// = {127,127};
    std::array<int, 2> m_search_size;// = {255,255};
    std::array<int, 2> m_output_size;// = {(g_search_size[0]-g_template_size[0]) / 8 + 1, (g_search_size[1] - g_template_size[1]) / 8 + 1};

    ACLTensor m_anchors;

    double m_scale_z;// = 0.0;
    double m_s_x;// = 0.0;
    int m_anchor_num;

    /////////////////
    std::string m_fonnx_file_path;
    std::string m_monnx_file_path;
    unsigned char * m_fmodel_data;
    int             m_fmodel_data_size;
    
    unsigned char * m_mmodel_data;
    int             m_mmodel_data_size;

    rknn_input_output_num m_f_io_num;
    rknn_tensor_attr      *m_f_input_attrs;
    rknn_tensor_attr      *m_f_output_attrs;

    int m_f_channel;
    int m_f_width;
    int m_f_height;
 
    rknn_input_output_num m_m_io_num;
    rknn_tensor_attr      *m_m_input_attrs;
    rknn_tensor_attr      *m_m_output_attrs;

    int m_m_channel;
    int m_m_width;
    int m_m_height;

    rknn_input * m_f_inputs;
    //rknn_output * m_f_outputs;

    rknn_input * m_m_inputs;
    //rknn_output * m_m_outputs;
    std::vector<float>    m_f_output0;
    std::vector<float>    m_f_output1;
    std::vector<float>    m_f_output2;

    rknn_context m_fctx;
    rknn_context m_mctx;

public:

    Tracker(const std::string &fmodel, const std::string &mmodel);
    ~Tracker();
    int init();
    int set_template(const cv::Mat &mat,cv::Rect bbox);
    ScoreRect track(const cv::Mat &mat);
private:
    ScoreRect postprocess(
         const ACLTensor &output_score,
         const ACLTensor &output_bbox,
         const cv::Mat &mat);

};

#endif // _INC_TRACKER_H

