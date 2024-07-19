#ifndef SEETA_CTRACKER_H
#define SEETA_CTRACKER_H

#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"



namespace seeta {

struct ScoreRect {
    float score;
    cv::Rect rect;
};


class CTracker
{
public:
    CTracker(const std::string &fmodel, const std::string &mmodel);
    ~CTracker();
    int init();
    int set_template(const cv::Mat &mat,cv::Rect bbox);
    ScoreRect track(const cv::Mat &mat);
private:
    class TrackerPrivate;
    TrackerPrivate *m_impl;

};


}

#endif
