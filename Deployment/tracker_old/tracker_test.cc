#include <string>
#include <chrono>
#include <iostream>

#include <ctracker.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


void usage()
{
    std::cout << " use:tracker_test template_model, search_model, timage, simage, x,y,w,h" << std::endl;
}

int main(int argc, char **argv)
{
     if(argc != 9)
     {
         usage();
         return 0;
     }
     seeta::CTracker track(argv[1], argv[2]);
     if (track.init() != 0)
     {
          std::cout << "init failed!" << std::endl;
          return -1;
     } 
     std::cout << "init ok" << std::endl;

     int x,y,w,h;
     x = atoi(argv[5]);
     y = atoi(argv[6]);
     w = atoi(argv[7]);
     h = atoi(argv[8]);

     cv::Mat tmat = cv::imread(argv[3]);
     cv::Mat smat = cv::imread(argv[4]);
     cv::Rect bbox = {x,y,w,h}; //{216,273,59,31};
     seeta::ScoreRect r;

     std::chrono::system_clock::time_point begintime = std::chrono::system_clock::now();
     for(int i=0; i<100; i++)
     {
         //std::cout << "begin set_template" << std::endl;
         track.set_template(tmat,bbox);

         //std::cout << "begin track" << std::endl;
         r = track.track(smat);
         //std::cout << "ret:" << r.ret << std::endl;
         std::cout << "score:" << r.score << std::endl;
         std::cout << "rect:" << r.rect.x << "," << r.rect.y << ","<< r.rect.width << "," << r.rect.height << std::endl;
     }

     std::chrono::system_clock::time_point endtime = std::chrono::system_clock::now();
     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - begintime);
     std::cout << "average:" << duration.count() / 100 << " ms" << std::endl;

     std::string oimg = "test_tmp.png";
     cv::rectangle(smat, r.rect, (0,255,0),3, 8, 0);
     cv::imwrite(oimg.c_str(), smat);

     return 0;
}


