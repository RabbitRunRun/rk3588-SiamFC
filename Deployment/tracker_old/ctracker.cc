#include <iostream> 
#include <cmath>
#include <climits>
#include <array>
#include <vector>
#include <chrono>


//#include "seeta_aip.h"
//#include "seeta_aip_opencv.h"
//#include "seeta_aip_image_io.h"
//#include "seeta_aip_affine.h"
//#include "seeta_aip_plot.h"
//#include "seeta_aip_plot_text.h"

#include "cann.h"
#include "utils.h"
#include "anchors.h"

//#include "resize_bicubic.h"


/////////////////////
//#include "RgaUtils.h"
//#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include "postprocess.h"
//#include "rga.h"
#include "rknn_api.h"


#include "ctracker.h"

#include "SeetaLANLock.h"
#include "hidden/SeetaLockFunction.h"
#include "hidden/SeetaLockVerifyLAN.h"



#define PERF_WITH_POST 1


#define PI 3.1415926
const float PENALTY_K = 0.04;
const float WINDOW_INFLUENCE = 0.44;
const float LR = 0.33;


////////////////////////////////
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{ 
     return ((float)qnt - (float)zp) * scale; 
}

template<typename T>
double average(const T *data, int len)
{
     double d = 0.0;
     for(int i=0; i<len; i++)
     {
         d += data[i]; 
     }
     return float(d / len);
}

template<typename T>
void nhwc2nchw(int N,int H, int W, int C , const T *data, T *dst)
{
    for(int n=0; n<N; n++)
    {
        for(int c=0; c<C; c++)
        {
            for(int h=0; h<H; h++)
            {
                for(int w=0; w<W; w++)
                {
                    int oldindex = n * H * W * C + h * W * C + w * C + c;
                    int newindex = n * C * H * W + c * H * W + h * W + w;
                    dst[newindex] = data[oldindex];
                }
            }
        }

    }
}

template<typename T>
void  nchw2nhwc(int N, int C, int H, int W, const T *input_data, T *output_data)
{
    // [N,C,H,W] -> [N,H,W,C]
    for (int n = 0; n < N; ++n)
    {
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                for (int c = 0; c < C; ++c)
                {
                    int old_index = n * C * H * W + c * H * W + h * W + w; // 原始索引值
                    int new_index = n * H * W * C + h * W * C + w * C + c; // 新索引值
                    output_data[new_index] = input_data[old_index];
                }
            }
        }
    }
}

static std::vector<float> CreateHannWindow(int num)
{
     std::vector<float> v;
     v.resize(num);
     for(int i=0; i<num; i++)
     {
          v[i] = 0.5 - 0.5 * cos(2 * PI * i / (num -1));
     }
     return v;
}


template<typename T>
static std::vector<T> outer(const std::vector<T>& a, const std::vector<T>& b)
{
    std::vector<T> result(a.size() * b.size());
    for(int i=0; i<a.size(); i++ )
    {
        for(int k=0; k<b.size(); k++)
        {
            result[i * b.size() + k] = b[k] * a[i];
        }
    }
    return result;
}


template<typename T>
static std::vector<T> tile(const std::vector<T> & a, unsigned int num)
{
    std::vector<T> res;
    res.resize(a.size() * num);
    for(int i=0; i<num; i++)
    {
        memcpy(res.data() + i * a.size(), a.data(), a.size() * sizeof(T));
    }
    return res;
}
////////////////////////////////




static ACLTensor generate_anchor(const std::array<int, 2> &score_size) {
    Anchors anchors(8, {0.33, 0.5, 1, 2, 3}, {8});
    /**
     * anchor = anchors.anchors
     * x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
     * anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
     */
    ACLTensor anchor(anchors.anchors.shape());  // .shape = [len(scales) * len(ratios), 4];
    {
        auto src = anchors.anchors.data();
        auto dst = anchor.data();
        auto n = anchor.shape(0);
        for (decltype(n) i = 0; i < n; ++i) {
            auto x1 = src[0], y1 = src[1], x2 = src[2], y2 = src[3];

            dst[0] = (x1 + x2) * 0.5f;
            dst[1] = (y1 + y2) * 0.5f;
            dst[2] = x2 - x1;
            dst[3] = y2 - y1;

            src += 4;
            dst += 4;
        }
    }
    /**
     * total_stride = anchors.stride
     * anchor_num = anchor.shape[0]
     */
    auto total_stride = anchors.stride;
    auto anchor_num = anchor.shape(0);
    /**
     * anchor = np.tile(anchor, score_size[0] * score_size[1]).reshape((-1, 4))
     */
    {
        auto m = anchor_num;
        auto n = score_size[0] * score_size[1];
        ACLTensor tile_anchor(std::vector<int64_t>({m * n, 4}));
        auto src = anchor.data();
        auto dst = tile_anchor.data();
        for (decltype(m) i = 0; i < m; ++i) {
            for (decltype(n) j = 0; j < n; ++j) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];

                dst += 4;
            }
            src += 4;
        }
        anchor = std::move(tile_anchor);
    }   // anchor.shape = [len(scales) * len(ratios) * score_size[0] * score_size[1], 4]
    /**
     * orix = - (score_size[0] / 2.) * total_stride
     * oriy = - (score_size[1] / 2.) * total_stride
     */
    auto orix = - (score_size[0] / 2.0f) * total_stride;
    auto oriy = - (score_size[1] / 2.0f) * total_stride;
    /**
     * xx, yy = np.meshgrid([orix + total_stride * dx for dx in range(score_size[0])],
     *                      [oriy + total_stride * dy for dy in range(score_size[1])])
     * xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
     *          np.tile(yy.flatten(), (anchor_num, 1)).flatten()
     * anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
     */
    {
        std::vector<float> x(score_size[0]); 
        std::vector<float> y(score_size[1]);
        for (size_t i = 0; i < x.size(); ++i) x[i] = orix + total_stride * i;
        for (size_t i = 0; i < y.size(); ++i) y[i] = oriy + total_stride * i;

        {
            auto n = anchor.shape(0);
            auto dst = &anchor.at(0, 0);
            auto xs = decltype(n)(x.size());
            for (decltype(n) i = 0; i < n; ++i) {
                *dst = x[i % xs];
                dst += 4;
            }
        }
        {
            auto n = anchor.shape(0);
            auto dst = &anchor.at(0, 1);
            auto xs = decltype(n)(x.size());
            auto ys = decltype(n)(y.size());
            for (decltype(n) i = 0; i < n; ++i) {
                *dst = y[(i / xs) % ys];
                dst += 4;
            }
        }
    }
    return anchor;
}

static float image_scale(SeetaAIPImageData img, const std::array<int, 2> &size) {
    auto h = float(img.height);
    auto w = float(img.width);
    auto s0 = float(size[0]);
    auto s1 = float(size[1]);

    auto scale = (h / w < s1 / s0) ? (s0 / w) : (s1 / h);
    return scale;
}

/*
static void fill_padding(
        SeetaAIPImageData img, const uint8_t (&color)[3],
        int x0, int x1, int y0, int y1) {
    auto w = int(img.width);
    auto h = int(img.height);
    struct Pixel { uint8_t rgb[3]; };
    auto c = (Pixel*)(&color[0]);
    {
        // top
        auto size = y0 * w;
        auto p = (Pixel*)img.data;
        for (int i = 0; i < size; ++i) {
            *p = *c;
            ++p;
        }
    }
    {
        // bottom
        auto size = (h - y1) * w;
        auto p = (Pixel*)img.data + y1 * w;
        for (int i = 0; i < size; ++i) {
            *p = *c;
            ++p;
        }
    }
    {
        // left
        auto step = x0;
        if (step) {
            auto size = y1 - y0;
            auto p = (Pixel*)img.data + y0 * w;
            auto jump = w - step;
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < step; ++j) {
                    *p = *c;
                    ++p;
                }
                p += jump;
            }
        }
    }
    {
        // right
        auto step = w - x1;
        if (step) {
            auto size = y1 - y0;
            auto p = (Pixel*)img.data + y0 * w + x1;
            auto jump = w - step;
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < step; ++j) {
                    *p = *c;
                    ++p;
                }
                p += jump;
            }
        }
    }
}
*/

/*
static seeta::aip::ImageData resize_pad(
        SeetaAIPImageData img, const std::array<int, 2> &size, float scale,
        const uint8_t (&color)[3],    
        float *ptop = nullptr, float *pleft = nullptr) {
    auto width = int(img.width * scale);
    auto height = int(img.height * scale);
    auto top    = (size[1] - height) / 2;
    auto bottom = (size[1] - height) - top;
    auto left   = (size[0] - width) / 2;
    auto right  = (size[0] - width) - left;

    float m[9] = {
            1 / scale, 0, -left / scale,
            0, 1 / scale, -top / scale,
            0, 0, 1
    };

    auto result = seeta::aip::affine_sample2d(1, m, img, 0, 0, size[0], size[1]);

    fill_padding(result, color, left, left + width, top, top + height);

    if (ptop) *ptop = top;
    if (pleft) *pleft = left;

    return result;
}
*/

/*
const SeetaAIPImageData image_pad(const SeetaAIPImageData& img, int left, int right, int top,int bottom, uint8_t(&c)[3])
{
        SeetaAIPImageData res;
        res.format = img.format;
        res.width = img.width + left + right;
        res.height = img.height + top + bottom;
        res.channels = img.channels;
        res.data = new unsigned char[res.width * res.height * res.channels];
        memset(res.data, 0, res.width * res.height * res.channels);

        // pad image
        //Image result(img.width() + 2 * w, img.height() + 2 * h, img.channels());
        //memset(result.data(), 0, result.count() * sizeof(Image::Datum));

        const unsigned char *iter_in_ptr = (const unsigned char *)img.data;
        int iter_in_step = img.width * img.channels;
        int copy_size = img.width * img.channels;
        int iter_size = img.height;
        unsigned char *iter_out_ptr = (unsigned char *)res.data + top * res.width * res.channels + left * res.channels;
        int iter_out_step = res.width * res.channels;

        for (int i = 0; i < iter_size; ++i, iter_in_ptr += iter_in_step, iter_out_ptr += iter_out_step)
        {
                memcpy(iter_out_ptr, iter_in_ptr, copy_size);
        }


        fill_padding(res, c, left, right, top,bottom);
        return res;
}

static SeetaAIPImageData image_crop(const SeetaAIPImageData& img, const Rect& rect)
{
        using namespace std;
        // Adjust rect
        Rect fixed_rect = rect;
        fixed_rect.w += fixed_rect.x;
        fixed_rect.h += fixed_rect.y;
        float tmpx = float(img.width - 1);
        fixed_rect.x = std::max<float>(0.0, std::min(tmpx, fixed_rect.x));

        float tmpy = float(img.height - 1);
        fixed_rect.y = std::max<float>(0.0, std::min(tmpy, fixed_rect.y));
        fixed_rect.w = std::max<float>(0.0, std::min(tmpx, fixed_rect.w));
        fixed_rect.h = std::max<float>(0.0, std::min(tmpy, fixed_rect.h));
        fixed_rect.w -= fixed_rect.x;
        fixed_rect.h -= fixed_rect.y;

        SeetaAIPImageData res;
        res.format = img.format;
        res.width = rect.w;
        res.height = rect.h;
        res.channels = img.channels;

        res.data = new unsigned char[res.width * res.height * res.channels];
        memset(res.data, 0, res.width * res.height * res.channels);
        //std::cout << "fixed_rect:" << fixed_rect.x << "," << fixed_rect.y << "," << fixed_rect.w << ", " << fixed_rect.h << std::endl; 

        
        const unsigned char *iter_in_ptr = (const unsigned char *)(img.data) + (int)(fixed_rect.y * img.width * img.channels + fixed_rect.x * img.channels);
        int iter_in_step = img.width * img.channels;
        int copy_size = int(fixed_rect.w * img.channels);
        int iter_size = int(fixed_rect.h);
        unsigned char *iter_out_ptr = (unsigned char *)res.data + std::max<int>(0, int(fixed_rect.y - rect.y)) * res.width * res.channels + std::max(0, int(fixed_rect.x - rect.x)) * res.channels;
        int iter_out_step = res.width * res.channels;

        for (int i = 0; i < iter_size; ++i, iter_in_ptr += iter_in_step, iter_out_ptr += iter_out_step)
        {
                memcpy(iter_out_ptr, iter_in_ptr, copy_size);
        }
        return res;
}
*/


static void mean(const cv::Mat & img, uint8_t (&c)[3]) {
    /*
    double sum[3] = {0, 0, 0};
    auto n = img.width * img.height;
    auto p = (uint8_t*)img.data;
    for (decltype(n) i = 0; i < n; ++i) {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
        p += 3;
    }
    c[0] = uint8_t(sum[0] / n);
    c[1] = uint8_t(sum[1] / n);
    c[2] = uint8_t(sum[2] / n);
    */

    c[0] = 0;
    c[1] = 0;
    c[2] = 0;
}


static std::vector<uint8_t> preprocess(cv::Mat &image) {
    std::chrono::system_clock::time_point begintime = std::chrono::system_clock::now();
    // convert to RGB + NCHW format
    //auto want_format = SEETA_AIP_FORMAT_CHW_U8BGR;
    //std::cout << "preprocess---" << std::endl;
    //auto want_format = SEETA_AIP_FORMAT_U8BGR;
    //seeta::aip::ImageData bgr_nchw;
    //uint8_t * ptr = (uint8_t *)image.data;

    /*
    if (image.format != want_format) {
        std::cout << "convert:" << std::endl;
        bgr_nchw = seeta::aip::convert(1, want_format, image);
        //image = bgr_nchw;
        ptr = (uint8_t *)bgr_nchw.data();
    }
    */
    //std::cout << "preprocess---2" << std::endl;

     
    //std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
    //ACLTensor data(std::vector<int64_t>({1, image.height, image.width, image.channels}));
    /*
    auto HWC = image.height * image.width * image.channels;
    auto src = ptr;//(const uint8_t*)image.data;
    auto dst = (float *)data.data();
    for (decltype(HWC) i = 0; i < HWC; ++i) {
        *dst = float(*src);
        ++src;
        ++dst;
    }
    */
    std::vector<uint8_t> vec;
    vec.resize(image.cols * image.rows * image.channels());
    memcpy(vec.data(), image.data, vec.size()); 
    std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);
    //auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(endtime2 - endtime);
    //std::cout << "preprocess:" << duration.count()  << std::endl;
    return vec;
}

std::vector<uint8_t> copy_acltensor(const ACLTensor &ten)
{
     //std::cout << "copy_acltensor:" << ten.count() << std::endl;
     std::vector<uint8_t> vec;
     vec.resize(ten.count());
     for(int i=0; i<ten.count(); i++)
     {
          vec[i] = uint8_t(ten.at(i));
     }

     //std::cout << "copy_acltensor 2:" << vec.size() << std::endl;
     return vec;
}


static float softmax(float a, float b) {
    /**
     * tmp = np.max(x, axis=1)
     * x -= tmp.reshape((x.shape[0], 1))
     * x = np.exp(x)
     * tmp = np.sum(x, axis=1)
     * x /= tmp.reshape((x.shape[0], 1))
     */
    auto max = std::max(a, b);
    a -= max;
    b -= max;
    a = std::exp(a);
    b = std::exp(b);
    auto sum = a + b;
    return b / sum; // get [:, 1] at outside
}

static ACLTensor convert_score(const ACLTensor &output) {
    // output.shape = [1, 10, 105, 105]
    /**
     * score = score.transpose(1, 2, 3, 0).reshape(2, -1).transpose(1, 0)
     * score = softmax(score)[:, 1]
     */
    auto n = output.count() / 2;
    ACLTensor score(std::vector<int64_t>({n}));
    auto pa = output.data();
    auto pb = pa + n;
    auto pc = score.data();
    for (decltype(n) i = 0; i < n; ++i) {
        *pc++ = softmax(*pa++, *pb++);
    }
    return score;
}

static int64_t argmax(const ACLTensor &x) {
    auto p = x.data();
    auto n = x.count();
    auto maxv = p[0];
    decltype(n) maxi = 0;
    for (decltype(n) i = 1; i < n; ++i) {
        if (p[i] > maxv) {
            maxi = i;
            maxv = p[i];
        }
    }
    return maxi;
}

/**
 * decode single bbox at i
 */
static Rect decode_bbox(const ACLTensor &output, const ACLTensor &anchor, int64_t i) {
    // output.shape = [1, 20, 105, 105]
    /**
     * delta = delta.transpose(1, 2, 3, 0).reshape(4, -1)
     * delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
     * delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
     * delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
     * delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
     */
    auto n = output.count() / 4;
    
    auto p = &output.at(i);
    auto q = &anchor.at(i, 0);
    auto d0 = p[0];
    auto d1 = p[n];
    auto d2 = p[n * 2];
    auto d3 = p[n * 3];
    auto a0 = q[0];
    auto a1 = q[1];
    auto a2 = q[2];
    auto a3 = q[3];

    auto x = d0 * a2 + a0;
    auto y = d1 * a3 + a1;
    auto w = std::exp(d2) * a2;
    auto h = std::exp(d3) * a3;

    return {x, y, w, h};
}

void convert_bbox(ACLTensor &output, const ACLTensor &anchor)
{
    int ncol = output.count() / 4;
    for(int i=0; i<ncol; i++)
    {
              output.at(i)  = output.at(i) * anchor.at(i,2) + anchor.at(i,0);
              output.at(ncol + i) = output.at(ncol+i) * anchor.at(i,3) + anchor.at(i,1);
              output.at(2 * ncol + i) = std::exp(output.at(2 * ncol + i)) * anchor.at(i,2);
              output.at(3 * ncol + i) = std::exp(output.at(3 * ncol + i)) * anchor.at(i,3);
    }
}


ACLTensor change(const ACLTensor &output)
{
    ACLTensor res(output.shape());
    for(int i=0; i<output.count(); i++)
    {
          res.at(i) = std::max<float>(output.at(i), 1.0 / output.at(i));
    }
    return res;
}



ACLTensor change2(float * width_data, float * height_data, int len, const std::array<int,2> &tmpsize)
{
    std::vector<int64_t> dims;
    dims.push_back(len);
    ACLTensor res(dims);
    float scale = float(tmpsize[0]) / tmpsize[1];
    for(int i=0; i<res.count(); i++)
    {
          float value = scale / (width_data[i] / height_data[i]);
          res.at(i) = std::max<float>(value, 1.0 / value);
    }
    return res;
}


ACLTensor sz(float *width_data, float * height_data, float scale, int len)
{
    std::vector<int64_t> dims;
    dims.push_back(len);
    ACLTensor res(dims);
    for(int i=0; i<len; i++)
    {
          float value = (width_data[i] + height_data[i] ) * 0.5;
          res.at(i) = std::sqrt((width_data[i] + value)  *  (height_data[i] + value));
          res.at(i) = res.at(i) / scale;
    }
    return res;
}

void bbox_clip(float & cx, float & cy, float &width, float & height, int image_width, int image_height)
{
    cx = std::max<float>(0.0, std::min<float>(cx, image_width));
    cy = std::max<float>(0.0, std::min<float>(cy, image_height));
    width = std::max<float>(10.0, std::min<float>(width, image_width));
    height = std::max<float>(10.0, std::min<float>(height, image_height));
}


std::ostream &operator<<(std::ostream &out, const ACLTensor &t) {
    std::ostringstream oss;
    oss << "float[";
    for (size_t i = 0; i < t.dims().size(); ++i) {
        if (i) oss << ", ";
        oss << t.dim(i);
    }
    oss << "]";
    return out << oss.str();
}



static cv::Mat get_subwindow(const cv::Mat &img, std::array<int,2> &pos, int model_sz, int original_sz, uint8_t (&c)[3])
{
     //std::chrono::system_clock::time_point begintime = std::chrono::system_clock::now();
     float value = (original_sz + 1) / 2;
     float context_xmin = floor(pos[0] - value + 0.5);
     float context_xmax = context_xmin + original_sz -1;
     float context_ymin = floor(pos[1] - value + 0.5);
     float context_ymax = context_ymin + original_sz -1;

     int left_pad = int(std::max<float>(0.0, -context_xmin));
     int top_pad = int(std::max<float>(0.0, -context_ymin)); 
     int right_pad = int(std::max<float>(0.0, context_xmax - img.cols + 1));
     int bottom_pad = int(std::max<float>(0.0, context_ymax - img.rows + 1));

     context_xmin = context_xmin + left_pad;
     context_xmax = context_xmax + left_pad;
     context_ymin = context_ymin + top_pad;
     context_ymax = context_ymax + top_pad;

     Rect rect;
     rect.x = int(context_xmin);
     rect.y = int(context_ymin);   //int(context_xmax + 1);
     rect.w = int(context_xmax + 1) - rect.x;// + 1;
     rect.h = int(context_ymax + 1) - rect.y;// + 1;

     //std::cout << "crop image:" << rect.x << "," << rect.y << "," << rect.w << ","<< rect.h << std::endl;

     //SeetaAIPImageData res;
     cv::Mat res;
     if (left_pad > 0 || top_pad > 0 || right_pad > 0 || bottom_pad > 0)
     {

          //std::cout << "-----get_sub_window-----image_pad" << std::endl;
          //SeetaAIPImageData imagepad = image_pad(img, left_pad, right_pad, top_pad, bottom_pad, c);

          //std::cout << "-----get_sub_window-----image_crop" << std::endl;
          //res = image_crop(imagepad, rect);// int(context_xmin),  int(context_xmax + 1), int(context_ymin), int(context_ymax + 1));
          //delete [] imagepad.data;
          cv::Mat padmat;
          cv::copyMakeBorder(img, padmat,top_pad, bottom_pad, left_pad,right_pad, cv::BORDER_CONSTANT, c[0]);
          res = padmat({rect.y,rect.h + rect.y},{rect.x,rect.x + rect.w}).clone(); 

     }else
     {

          //std::cout << "-----get_sub_window-----image_crop 2" << std::endl;
          //res = image_crop(img, rect);//int(context_xmin),  int(context_xmax + 1), int(context_ymin), int(context_ymax + 1));
          
          //std::cout << "crop image:" << rect.x << "," << rect.y << "," << rect.w << ","<< rect.h << std::endl;
          res = img({rect.y,rect.h + rect.y},{rect.x,rect.x + rect.w}).clone(); 
          //cv::Mat cropimage(img.height, img.width, CV_8UC3);
          //memcpy(cropimage.data, img.data, img.height * img.width * 3);
          //cv::Mat dmat = cropimage({rect.y,rect.h + rect.y},{rect.x,rect.x + rect.w}).clone();
          //res.data = new unsigned char[int(rect.w) * int(rect.h) * 3]; 
          //memcpy(res.data, dmat.data, dmat.cols * dmat.rows * 3);
          //res.format = img.format;
          //res.width = dmat.cols;
          //res.height = dmat.rows;
          //res.channels = 3;

     }

     //ACLTensor ten;
     std::vector<uint8_t> ten;
     if (model_sz != original_sz)
     {
          //SeetaAIPImageData s;
          //s.format = res.format;
          //s.width = model_sz;
          //s.height = model_sz;
          //s.channels = res.channels;

          //s.data = new unsigned char[s.width * s.height * s.channels];
          //memset(s.data, 0, s.width * s.height * s.channels);

          //std::chrono::system_clock::time_point begintime2 = std::chrono::system_clock::now();
          //std::cout << "-----get_sub_window-----resize_bicubic" << std::endl;
          //resize_bicubic<uint8_t>((const unsigned char *)res.data,res.width,res.height,res.channels, (unsigned char *)s.data,s.width,s.height);

          //delete [] res.data; 
          //cv::Mat srcmat(cv::Size(res.height,res.width), CV_8UC3);
          //memcpy(srcmat.data, res.data, res.height * res.width * 3); 

          cv::Mat dstmat;//({s.height,s.width},cv::CV::8UC3);
          cv::resize(res, dstmat, cv::Size(model_sz, model_sz),0,0);
          //cv::resize(srcmat, dstmat, cv::Size(s.height, s.width),0,0);
          //memcpy(s.data, dstmat.data, s.height * s.width * 3);
          //if(image_resize((uint8_t *)(res.data), res.width, res.height, (uint8_t *)(s.data), s.width, s.height) < 0)
          //{
              //std::cout << "-----get_sub_window-----image_resize ok" << std::endl;
          //    return ten;
          //}
          //std::chrono::system_clock::time_point endtime2 = std::chrono::system_clock::now();

          //auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(endtime2 - begintime2);
          //std::cout << "-----get_sub_window-----preprocess," << duration2.count()  << std::endl;
          //ten = preprocess(dstmat);
          return dstmat;
          //delete [] s.data; 
     }else 
     {

          //std::cout << "-----get_sub_window-----preprocess 2" << std::endl;
          //ten = preprocess(res);    
          //return res;
          return res;
     }

     //std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
     //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);
     //std::cout << "get_subwindow:" << duration.count() << std::endl;
     //delete [] res.data; 
     //return ten;
}



///////////////////////
static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
  FILE* fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}


///////////////////////////////
/////////////////////////////////
namespace seeta {


class CTracker::TrackerPrivate {
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

    TrackerPrivate(const std::string &fmodel, const std::string &mmodel);
    ~TrackerPrivate();
    int init();
    int set_template(const cv::Mat &mat,cv::Rect bbox);
    ScoreRect track(const cv::Mat &mat);
private:
    ScoreRect postprocess(
         const ACLTensor &output_score,
         const ACLTensor &output_bbox,
         const cv::Mat &mat);

};


CTracker::TrackerPrivate::TrackerPrivate(const std::string &fmodel, const std::string &mmodel)
{
     m_anchor_num = 5;
     m_channel_average[0] = 0;
     m_channel_average[1] = 0;
     m_channel_average[1] = 0;

     m_template_size[0] = 127;
     m_template_size[1] = 127;

     m_search_size[0] = 255;
     m_search_size[1] = 255;

     m_output_size[0] = (m_search_size[0]-m_template_size[0]) / 8 + 1;
     m_output_size[1] = (m_search_size[1] - m_template_size[1]) / 8 + 1;

     m_scale_z = 0.0;
     m_s_x = 0.0;

     m_fonnx_file_path = fmodel;
     m_monnx_file_path = mmodel;

     m_fmodel_data = NULL;
     m_mmodel_data = NULL;

     m_f_input_attrs = NULL;
     m_m_input_attrs = NULL;

     m_f_output_attrs = NULL;
     m_m_output_attrs = NULL;

     m_f_inputs = NULL;
     m_m_inputs = NULL;

     //m_f_output0.resize(1 * 256 * 7 * 7);
     m_f_output0.resize(1 * 256 * 6 * 6);

     m_f_output1.resize(1 * 256 * 7 * 7);
     m_f_output2.resize(1 * 256 * 7 * 7);
     memset(m_f_output0.data(), 0, sizeof(m_f_output0.size()) * sizeof(float));
     memset(m_f_output1.data(), 0, sizeof(m_f_output1.size()) * sizeof(float));
     memset(m_f_output2.data(), 0, sizeof(m_f_output2.size()) * sizeof(float));
}


CTracker::TrackerPrivate::~TrackerPrivate()
{
    rknn_destroy(m_fctx);
    rknn_destroy(m_mctx);

    if (m_fmodel_data) {
       free(m_fmodel_data);
    }

    if (m_mmodel_data) {
       free(m_mmodel_data);
    }


    if(m_f_input_attrs)
    {
        delete [] m_f_input_attrs;
    }
    if(m_m_input_attrs)
    {
        delete [] m_m_input_attrs;
    }
    if(m_f_output_attrs)
    {
        delete [] m_f_output_attrs;
    }
    if(m_m_output_attrs)
    {
        delete [] m_m_output_attrs;
    }

    if(m_f_inputs)
    {
        delete [] m_f_inputs;
    }
    if(m_m_inputs)
    {
        delete [] m_m_inputs;
    }
    //////////////////////////////
}

//return 0,ok; -1,failed
int CTracker::TrackerPrivate::init()
{
    if (m_template_size[0] < 160)
    {
        m_output_size[0] = m_search_size[0] / 8 -7 + 1;
        m_output_size[1] = m_search_size[1] / 8 -7 + 1;
    }


    m_output_size[0] = 17;
    m_output_size[1] = 17;

    m_anchors = generate_anchor(m_output_size);

    std::vector<float> hanning = CreateHannWindow(m_output_size[0]);
    std::vector<float> outerwindow = outer(hanning, hanning);
    m_window = tile<float>(outerwindow, 5);


    printf("Loading mode...\n");
    m_fmodel_data_size = 0;
    m_fmodel_data   = load_model(m_fonnx_file_path.c_str(), &m_fmodel_data_size);
    int ret = rknn_init(&m_fctx, m_fmodel_data, m_fmodel_data_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init load model:%s error ret=%d\n", m_fonnx_file_path.c_str(), ret);
        return -1;
    }

    m_mmodel_data_size = 0;
    m_mmodel_data = load_model(m_monnx_file_path.c_str(), &m_mmodel_data_size);
    ret = rknn_init(&m_mctx, m_mmodel_data, m_mmodel_data_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init load model:%s error ret=%d\n", m_monnx_file_path.c_str(), ret);
        return -1;
    }

    rknn_sdk_version fversion;
    ret = rknn_query(m_fctx, RKNN_QUERY_SDK_VERSION, &fversion, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("template rknn_query RKNN_QUERY_SDK_VERSION, error ret=%d\n", ret);
        return -1;
    }
    printf("template sdk version: %s driver version: %s\n", fversion.api_version, fversion.drv_version);

    //rknn_input_output_num f_io_num;
    ret = rknn_query(m_fctx, RKNN_QUERY_IN_OUT_NUM, &m_f_io_num, sizeof(m_f_io_num));
    if (ret < 0) {
        printf("tempalte query input_output_num error ret=%d\n", ret);
        return -1;
    }
    printf("template input num: %d, output num: %d\n", m_f_io_num.n_input, m_f_io_num.n_output);

    //rknn_tensor_attr f_input_attrs[f_io_num.n_input];
    m_f_input_attrs = new rknn_tensor_attr[m_f_io_num.n_input];
    memset(m_f_input_attrs, 0, sizeof(rknn_tensor_attr) * m_f_io_num.n_input );
    for (int i = 0; i < m_f_io_num.n_input; i++) {
        m_f_input_attrs[i].index = i;
        ret = rknn_query(m_fctx, RKNN_QUERY_INPUT_ATTR, &(m_f_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("template query_input_attr error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(m_f_input_attrs[i]));
    }

    m_f_output_attrs = new rknn_tensor_attr[m_f_io_num.n_output]; 
    memset(m_f_output_attrs, 0, sizeof(rknn_tensor_attr) * m_f_io_num.n_output);
    for (int i = 0; i < m_f_io_num.n_output; i++) {
        m_f_output_attrs[i].index = i;
        ret = rknn_query(m_fctx, RKNN_QUERY_OUTPUT_ATTR, &(m_f_output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(m_f_output_attrs[i]));
    }

    m_f_channel = 3;
    m_f_width   = 0;
    m_f_height  = 0;
    if (m_f_input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        m_f_channel = m_f_input_attrs[0].dims[1];
        m_f_height  = m_f_input_attrs[0].dims[2];
        m_f_width   = m_f_input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        m_f_height  = m_f_input_attrs[0].dims[1];
        m_f_width   = m_f_input_attrs[0].dims[2];
        m_f_channel = m_f_input_attrs[0].dims[3];
    }

    printf("template model input height=%d, width=%d, channel=%d\n", m_f_height, m_f_width, m_f_channel);

    /////////////////////////////////////////////

    rknn_sdk_version mversion;
    ret = rknn_query(m_mctx, RKNN_QUERY_SDK_VERSION, &mversion, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("search rknn_query RKNN_QUERY_SDK_VERSION, error ret=%d\n", ret);
        return -1;
    }
    printf("search sdk version: %s driver version: %s\n", mversion.api_version, mversion.drv_version);

    //rknn_input_output_num f_io_num;
    ret = rknn_query(m_mctx, RKNN_QUERY_IN_OUT_NUM, &m_m_io_num, sizeof(m_m_io_num));
    if (ret < 0) {
        printf("search query input_output_num error ret=%d\n", ret);
        return -1;
    }
    printf("search input num: %d, output num: %d\n", m_m_io_num.n_input, m_m_io_num.n_output);

    //rknn_tensor_attr f_input_attrs[f_io_num.n_input];
    m_m_input_attrs = new rknn_tensor_attr[m_m_io_num.n_input];
    memset(m_m_input_attrs, 0, sizeof(rknn_tensor_attr) * m_m_io_num.n_input );
    for (int i = 0; i < m_m_io_num.n_input; i++) {
        m_m_input_attrs[i].index = i;
        ret = rknn_query(m_mctx, RKNN_QUERY_INPUT_ATTR, &(m_m_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("search query_input_attr error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(m_m_input_attrs[i]));
    }

    //std::cout << "search output attr:" << std::endl;
    //rknn_tensor_attr f_output_attrs[f_io_num.n_output];
    m_m_output_attrs = new rknn_tensor_attr[m_m_io_num.n_output]; 
    memset(m_m_output_attrs, 0, sizeof(rknn_tensor_attr) * m_m_io_num.n_output);
    for (int i = 0; i < m_m_io_num.n_output; i++) {
        m_m_output_attrs[i].index = i;
        ret = rknn_query(m_mctx, RKNN_QUERY_OUTPUT_ATTR, &(m_m_output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(m_m_output_attrs[i]));
    }

    m_m_channel = 3;
    m_m_width   = 0;
    m_m_height  = 0;
    if (m_m_input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("search model is NCHW input fmt\n");
        m_m_channel = m_m_input_attrs[0].dims[1];
        m_m_height  = m_m_input_attrs[0].dims[2];
        m_m_width   = m_m_input_attrs[0].dims[3];
    } else {
        printf("search model is NHWC input fmt\n");
        m_m_height  = m_m_input_attrs[0].dims[1];
        m_m_width   = m_m_input_attrs[0].dims[2];
        m_m_channel = m_m_input_attrs[0].dims[3];
    }

    printf("search model input height=%d, width=%d, channel=%d\n", m_m_height, m_m_width, m_m_channel);
    ////////////////////////////////
  
    //rknn_input f_inputs[1];
    m_f_inputs = new rknn_input[m_f_io_num.n_input];
    memset(m_f_inputs, 0, sizeof(rknn_input) * m_f_io_num.n_input);
    m_f_inputs[0].index        = 0;
    m_f_inputs[0].type         = RKNN_TENSOR_UINT8;//m_f_input_attrs[0].type; //RKNN_TENSOR_UINT8;
    m_f_inputs[0].size         = m_f_width * m_f_height * m_f_channel;
    m_f_inputs[0].fmt          = RKNN_TENSOR_NHWC;//RKNN_TENSOR_NCHW;//m_f_input_attrs[0].fmt;//RKNN_TENSOR_NCHW;
    m_f_inputs[0].pass_through = 0;
  

    m_m_inputs = new rknn_input[m_m_io_num.n_input];
    //rknn_input m_inputs[m_io_num.n_input];
    memset(m_m_inputs, 0, sizeof(rknn_input) * m_m_io_num.n_input);
    for(int i=0; i<m_m_io_num.n_input; i++)
    {
        m_m_inputs[i].index        = i;
        m_m_inputs[i].type         = RKNN_TENSOR_FLOAT32;//m_m_input_attrs[i].type; //RKNN_TENSOR_UINT8;
        //if(i==3)
        if(i==1)
        {
            m_m_inputs[i].type         = RKNN_TENSOR_UINT8;//m_m_input_attrs[i].type; //RKNN_TENSOR_UINT8;
        }
        //f_inputs[i].size         = f_width * f_height * f_channel;
        m_m_inputs[i].fmt          = RKNN_TENSOR_NHWC;//m_m_input_attrs[i].fmt;//RKNN_TENSOR_NCHW;
        m_m_inputs[i].pass_through = 0;
    } 

    /*
    m_m_outputs = new rknn_output[m_m_io_num.n_output];
    //rknn_output m_outputs[m_io_num.n_output];
    memset(m_m_outputs, 0, sizeof(rknn_output) * m_m_io_num.n_output);
    for (int i = 0; i < m_m_io_num.n_output; i++) 
    {
        m_m_outputs[i].want_float = 1;
    }
    */
    return 0;
}



int CTracker::TrackerPrivate::set_template(const cv::Mat &mat,cv::Rect bbox)
{
    std::chrono::system_clock::time_point begintime = std::chrono::system_clock::now();
    //std::cout << "----set_template---w:" << mat.cols << ", h" << mat.rows  << std::endl;
    m_center_pos[0] = bbox.x + (bbox.width -1) / 2;
    m_center_pos[1] = bbox.y + (bbox.height -1) / 2;
    m_size[0] = bbox.width;
    m_size[1] = bbox.height;

    int w_z = m_size[0] + 0.5 * (m_size[0] + m_size[1]);
    int h_z = m_size[1] + 0.5 * (m_size[0] + m_size[1]);

    double s_z = round(std::sqrt(w_z * h_z));
    m_scale_z = m_template_size[0] / s_z;
    m_s_x = s_z * (m_search_size[0] / m_template_size[0]);

    /*
    SeetaAIPImageData s;
    s.format = SEETA_AIP_FORMAT_U8BGR;//res.format;
    s.width = mat.cols;
    s.height = mat.rows;
    s.channels = mat.channels();
    s.data = mat.data;//new unsigned char[s.weight * s.height * s.channels];
    */
    //memset(s.data, 0, s.weight * s.height * s.channels);
    //mean(s, m_channel_average); 
    
    mean(mat, m_channel_average); 
    //std::cout << "----set_template-- get_subwindow-" << std::endl;
    //ACLTensor z_crop = get_subwindow(s, m_center_pos, m_template_size[0], s_z, m_channel_average); 
    //std::cout << "z_crop:" << z_crop.shape(0) << "," << z_crop.shape(1) << "," << z_crop.shape(2) << ", " << z_crop.shape(3) <<  std::endl;
    

    //std::vector<uint8_t> vec = copy_acltensor(z_crop);
    //std::vector<uint8_t> vec = get_subwindow(mat, m_center_pos, m_template_size[0], s_z, m_channel_average);

    cv::Mat dmat = get_subwindow(mat, m_center_pos, m_template_size[0], s_z, m_channel_average);

    //std::cout << "----set_template-- get_subwindow-dd" << std::endl;
    //cv::imwrite("template_input.png",dmat);
     /*
    SeetaAIPImageData test;
    test.format = SEETA_AIP_FORMAT_U8BGR;//res.format;
    test.width = z_crop.shape(2);
    test.height = z_crop.shape(1);
    test.channels = z_crop.shape(3);
    test.data = vec.data();//new unsigned char[s.weight * s.height * s.channels];
    seeta::aip::imwrite("test1.png",test);
    */
    //cv::Mat testmat(z_crop.shape(1), z_crop.shape(2),CV_8UC3);
    //memcpy(testmat.data ,vec.data(), vec.size());//new unsigned char[s.weight * s.height * s.channels];
    //cv::imwrite("test1.png", testmat);


    //m_f_inputs[0].buf = (void*)vec.data();//(void *)z_crop.data();
    //m_f_inputs[0].size = vec.size();

    m_f_inputs[0].buf = (void*)dmat.data;//(void *)z_crop.data();
    m_f_inputs[0].size = dmat.cols * dmat.rows * dmat.channels();

    /*
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(rknn_input));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;//m_f_input_attrs[0].type; //RKNN_TENSOR_UINT8;
    inputs[0].size         = vec.size(); //m_f_width * m_f_height * m_f_channel;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;//RKNN_TENSOR_NCHW;//m_f_input_attrs[0].fmt;//RKNN_TENSOR_NCHW;
    inputs[0].pass_through = 0;
    inputs[0].buf = (void*)vec.data();//(void *)z_crop.data();
    */
    //rknn_inputs_set(m_fctx, m_f_io_num.n_input, m_f_inputs);
    
    rknn_inputs_set(m_fctx, 1, m_f_inputs);
    //rknn_inputs_set(m_fctx, 1, inputs);

    //rknn_output outputs[3];
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; 
    //outputs[1].want_float = 1; 
    //outputs[2].want_float = 1; 

    //std::cout << "----set_template-- rknn_run-" << std::endl;
    int ret = rknn_run(m_fctx, NULL);
    if(ret != RKNN_SUCC)
    {
         std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);

         printf("rknn_run failed, ret:%d, spend time:%ld ms\n",ret, duration.count());
         return -1;
    }

    //std::cout << "----set_template-- rknn_run  ok:----" << m_f_io_num.n_output << std::endl;
    //ret = rknn_outputs_get(m_fctx, m_f_io_num.n_output, m_f_outputs, NULL);

    //ret = rknn_outputs_get(m_fctx, 3, m_f_outputs, NULL);
    //ret = rknn_outputs_get(m_fctx, 3, outputs, NULL);
    ret = rknn_outputs_get(m_fctx, 1, outputs, NULL);
    if (ret != RKNN_SUCC)
    {

         std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);
         printf("rknn_outputs_get failed, ret:%d, spend time:%ld ms\n",ret, duration.count());
         return -1;
    }


    nchw2nhwc<float>(1, 256,6, 6, (float *)(outputs[0].buf), m_f_output0.data());
    //nchw2nhwc<float>(1, 256,7, 7, (float *)(outputs[0].buf), m_f_output0.data());
    //nchw2nhwc<float>(1, 256,7, 7, (float *)(outputs[1].buf), m_f_output1.data());
    //nchw2nhwc<float>(1, 256,7, 7, (float *)(outputs[2].buf), m_f_output2.data());

    //std::cout << "output:" << outputs[0].size << "," << outputs[1].size << ", " << outputs[2].size << std::endl;
    //std::cout << "output:" << m_f_outputs[0].size << "," << m_f_outputs[1].size << ", " << m_f_outputs[2].size << std::endl;
    //rknn_outputs_release(m_fctx, 3, outputs);
    rknn_outputs_release(m_fctx, 1, outputs);

    std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);
    printf("set_template ok, spend time:%ld ms\n", duration.count());
    return 0;
}

ScoreRect CTracker::TrackerPrivate::track(const cv::Mat &mat)
{

    std::chrono::system_clock::time_point begintime = std::chrono::system_clock::now();
    //std::cout << "----track-- " << std::endl;
    /*
    SeetaAIPImageData s;
    s.format = SEETA_AIP_FORMAT_U8BGR;//res.format;
    s.width = mat.cols;
    s.height = mat.rows;
    s.channels = mat.channels();
    s.data = mat.data;//new unsigned char[s.weight * s.height * s.channels];
    */
    //std::cout << "----track--get_subwindow " << std::endl;
    //ACLTensor x_crop = get_subwindow(s, m_center_pos, m_search_size[0], round(m_s_x), m_channel_average); 
    //std::vector<uint8_t> vec = copy_acltensor(x_crop);

    //std::vector<uint8_t> vec = get_subwindow(s, m_center_pos, m_search_size[0], round(m_s_x), m_channel_average);
    cv::Mat dmat = get_subwindow(mat, m_center_pos, m_search_size[0], round(m_s_x), m_channel_average);
    //cv::imwrite("track_input.png",dmat);
    //std::cout << "x_crop:" << x_crop.shape(0) << "," << x_crop.shape(1) << "," << x_crop.shape(2) << "," << x_crop.shape(3) << std::endl;
    //std::cout << "tempate output 0 size:" << m_f_outputs[0].size << std::endl;
    //std::cout << "tempate output 1 size:" << m_f_outputs[1].size << std::endl;
    //std::cout << "tempate output 2 size:" << m_f_outputs[2].size << std::endl;

    //std::vector<float> input_buf1;
    //input_buf1.resize(m_f_outputs[0].size / 4);
    //input_buf1.resize(m_f_outputs[0].size / 4);
    //nchw2nhwc<float>(1, 256,7, 7, (float *)(m_f_outputs[0].buf), input_buf1.data());

    //std::vector<uint8_t> input_buf1;
    //input_buf1.resize(m_f_outputs[0].size);
    //nchw2nhwc<uint8_t>(1, 256,7, 7, (uint8_t *)(m_f_outputs[0].buf), input_buf1.data());
    
    //std::vector<float> input_buf2;
    //input_buf2.resize(m_f_outputs[1].size / 4);
    //nchw2nhwc<float>(1, 256,7, 7, (float *)(m_f_outputs[1].buf), input_buf2.data());

    //std::vector<uint8_t> input_buf2;
    //input_buf2.resize(m_f_outputs[1].size);
    //nchw2nhwc<uint8_t>(1, 256,7, 7, (uint8_t *)(m_f_outputs[1].buf), input_buf2.data());

    //std::vector<uint8_t> input_buf3;
    //input_buf3.resize(m_f_outputs[2].size);
    //nchw2nhwc<uint8_t>(1, 256,7, 7, (uint8_t *)(m_f_outputs[2].buf), input_buf3.data());
    //std::vector<float> input_buf3;
    //input_buf3.resize(m_f_outputs[2].size / 4);
    //nchw2nhwc<float>(1, 256,7, 7, (float *)(m_f_outputs[2].buf), input_buf3.data());

    m_m_inputs[0].buf = m_f_output0.data();//input_buf1.data(); //m_f_outputs[0].buf;
    m_m_inputs[0].size = m_f_output0.size() * sizeof(float);

    //m_m_inputs[1].buf = m_f_output1.data();//input_buf1.data(); //m_f_outputs[0].buf;
    //m_m_inputs[1].size = m_f_output1.size() * sizeof(float);
    
    //m_m_inputs[2].buf = m_f_output2.data();//input_buf1.data(); //m_f_outputs[0].buf;
    //m_m_inputs[2].size = m_f_output2.size() * sizeof(float);

    //m_m_inputs[3].buf = (void *)vec.data(); //x_crop.data(); 
    //m_m_inputs[3].size = vec.size();//x_crop.count();//x_crop.count();
   
    //m_m_inputs[1].buf = (void *)vec.data(); //x_crop.data(); 
    //m_m_inputs[1].size =  vec.size();; 
    m_m_inputs[1].buf = (void *)dmat.data; //x_crop.data(); 
    m_m_inputs[1].size =  dmat.rows * dmat.cols * 3;//vec.size();; 

    //for(int i=0; i<50; i++)
    //    std::cout << "," << deqnt_affine_to_f32(input_buf1[i], 7, 0.028732);
    //std::cout << std::endl;

    //for(int i=0; i<50; i++)
    //    std::cout << "," << (input_buf1[i]);
    //std::cout << std::endl;
    /*
    for(int i=0; i<50; i++)
        std::cout << "," << int(input_buf2[i]);
    std::cout << std::endl;
    for(int i=0; i<50; i++)
        std::cout << "," << int(input_buf3[i]);
    std::cout << std::endl;
    */
 
    //std::cout << "----track--rknn_inputs_set " << std::endl;
    //rknn_inputs_set(m_mctx, 4, m_m_inputs);
   
    rknn_inputs_set(m_mctx, 2, m_m_inputs);
    ScoreRect res;


    std::chrono::system_clock::time_point preendtime = std::chrono::system_clock::now();
    auto pre_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preendtime - begintime);

    //std::cout << "preprocess:" << pre_duration.count() << std::endl;
 
    //std::cout << "----track--rknn_run " << std::endl;
    int ret = rknn_run(m_mctx, NULL);
    if(ret != RKNN_SUCC)
    {
         std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);
         //res.ret = -1;
         printf("rknn_run failed, ret:%d, spend time:%d ms\n",ret, duration.count());
         return res;
    }


    rknn_output outputs[2];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; 
    outputs[1].want_float = 1; 
    
    //std::cout << "----track--rknn_outputs_get " << std::endl;
    ret = rknn_outputs_get(m_mctx, 2, outputs, NULL);
    if (ret != RKNN_SUCC)
    {
         std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);
         //res.ret = -1;
         printf("rknn_outputs_get failed, ret:%d, spend time:%ld ms\n",ret, duration.count());
         return res;
    }
    //printf("track ok\n");

    /*
    dump_tensor_attr(&(m_m_output_attrs[0]));
    std::cout << std::endl;
    dump_tensor_attr(&(m_m_output_attrs[1]));
    std::cout << "m_m_outputs[0].size:" << m_m_outputs[0].size << std::endl;
    std::cout << "m_m_outputs[1].size:" << m_m_outputs[1].size << std::endl;
    */
    //std::vector<int64_t> dims_0 = {1,10,25,25};
    std::vector<int64_t> dims_0 = {1,10,17,17};

    //dims_0.resize(4);   
    //std::cout << "dims:" << m_m_output_attrs[0].dims[0]  << "," << m_m_output_attrs[0].dims[1] << "," <<m_m_output_attrs[0].dims[2] << "," << m_m_output_attrs[0].dims[3] << std::endl; 
 

    //std::vector<int64_t> dims_1 = {1,20,25,25};
    std::vector<int64_t> dims_1 = {1,20,17,17};

    //std::cout << "m_m_outputs[0].size:" << outputs[0].size << std::endl;

    ACLTensor output0(dims_0);
    memcpy(output0.data(), outputs[0].buf, outputs[0].size);
 
    ACLTensor output1(dims_1);

    //std::cout << "m_m_outputs[1].size:" << outputs[1].size << std::endl;
    memcpy(output1.data(), outputs[1].buf, outputs[1].size);

    //printf("----track--- postprocess \n");

    std::chrono::system_clock::time_point pbegintime = std::chrono::system_clock::now();
    ScoreRect r = postprocess(output0, output1, mat);
    //r.ret = 0;
    std::chrono::system_clock::time_point pendtime = std::chrono::system_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(pendtime - pbegintime);

    //std::cout << "postprocess:" << duration2.count() << std::endl;
    //rknn_outputs_release(m_mctx, m_m_io_num.n_output, m_m_outputs);
    //printf("----track--- rknn_outputs_release \n");
    rknn_outputs_release(m_mctx, 2, outputs);

    //ScoreRect r ;//= postprocess(output0, output1, mat);

    std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);

    printf("track success, spend time:%ld ms\n", duration.count());
    return r;
}


ScoreRect CTracker::TrackerPrivate::postprocess(
        const ACLTensor &output_score, 
        const ACLTensor &output_bbox,
        const cv::Mat &mat)
{
    ACLTensor score = convert_score(output_score);
    //int cols = score.count() / 4;
    //std::cout << "score max:" << tmp << std::endl;
    //std::cout << "m_scale_z:" << m_scale_z << std::endl;
    //std::cout << "m_size:" << m_size[0] << "," << m_size[1] << std::endl;
    //std::cout << "m_s_x :" << m_s_x << std::endl;
    //std::cout << "centerpos:" << m_center_pos[0] << ", " << m_center_pos[1] << std::endl;



    ACLTensor pred_bbox(output_bbox.shape());
    memcpy(pred_bbox.data(), output_bbox.data(), output_bbox.bytes());


    convert_bbox(pred_bbox, m_anchors);

    int cols = pred_bbox.count() / 4;

    float tmp_w = m_size[0] * m_scale_z;
    float tmp_h = m_size[1] * m_scale_z;
    float tmp_pad = (tmp_w + tmp_h) * 0.5;
    float tmp_sz = std::sqrt((tmp_w + tmp_pad) * (tmp_h + tmp_pad));

    ACLTensor s_c = change(sz(pred_bbox.data() + 2 * cols, pred_bbox.data() + 3 * cols, tmp_sz, cols));

    ACLTensor r_c = change2(pred_bbox.data() + 2 * cols, pred_bbox.data() + 3 * cols, cols, m_size);

    ACLTensor penalty(r_c.shape());
    for(int i=0; i<r_c.count(); i++)
    {
        penalty.at(i) = std::exp(-(r_c.at(i) * s_c.at(i) - 1) * PENALTY_K); 
    }

    ACLTensor pscore(score.shape());
    for(int i=0; i<pscore.count(); i++)
    {
        pscore.at(i) = score.at(i) * penalty.at(i); 
        pscore.at(i) = pscore.at(i) * (1 - WINDOW_INFLUENCE) + m_window[i] * WINDOW_INFLUENCE;
    }

    auto best_idx = argmax(pscore);

    //std::cout << "best_idex:" << best_idx << std::endl;

    std::array<float, 4> bbox;
    for (int i=0; i< 4; i++)
    {
          bbox[i] = pred_bbox.at(i * cols + best_idx) / m_scale_z;
          //bbox[i] = pred_bbox.at(i, best_idx) / m_scale_z;
    } 

    float lr = penalty.at(best_idx) * score.at(best_idx) * LR;

    float cx = bbox[0] + m_center_pos[0];
    float cy = bbox[1] + m_center_pos[1];
    float width = m_size[0] * (1 - lr) + bbox[2] * lr;
    float height = m_size[1] * (1 - lr) + bbox[3] * lr;

    bbox_clip(cx, cy, width, height, mat.cols, mat.rows);

    m_center_pos[0] = cx;
    m_center_pos[1] = cy;

    m_size[0] = width;
    m_size[1] = height;

     
    //auto best_score = score.at(best_idx);

    ScoreRect result;
    result.score = score.at(best_idx);
    result.rect.x = cx - width / 2;
    result.rect.y = cy - height / 2;
    result.rect.width = width;
    result.rect.height = height;
    //result.ret = 0;
    return result;
}


CTracker::CTracker(const std::string &fmodel, const std::string &mmodel)
{
    SeetaLock_VerifyLAN verify(1006);
    SeetaLockSafe_call(&verify);

    SeetaLock_Verify_Check_Instances verify_instances("seeta.tracker");
    SeetaLockSafe_call(&verify_instances);
    m_impl = NULL;
    m_impl = new TrackerPrivate(fmodel, mmodel);
}

CTracker::~CTracker()
{
    if(m_impl)
        delete m_impl;
}

int CTracker::init()
{
    return m_impl->init();
}

int CTracker::set_template(const cv::Mat &mat,cv::Rect bbox)
{
    return m_impl->set_template(mat, bbox);
}

ScoreRect CTracker::track(const cv::Mat &mat)
{
    return m_impl->track(mat);
}


//////////////////
}
