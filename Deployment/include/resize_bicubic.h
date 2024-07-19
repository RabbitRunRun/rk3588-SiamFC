#ifndef _INC_RESIZE_BICUBIC_H
#define _INC_RESIZE_BICUBIC_H

template<typename T>
inline void resize_bicubic(
        const T *src_im, int src_width, int src_height, int channels,
        T *dst_im, int dst_width, int dst_height) {
    double scale_x = (double) src_width / dst_width;
    double scale_y = (double) src_height / dst_height;

    int srcrows = src_width * channels;
    int dstrows = dst_width * channels;
    
    for (int j = 0; j < dst_height; ++j) {
        double fy = (double) ((j + 0.5) * scale_y - 0.5);
        int sy = int(floor(fy));
        fy -= sy;
        //sy = std::min(sy, src_height - 3);
        //sy = std::max(1, sy);
        if (sy < 1) {
            fy = 0;
            sy = 1;
        }

        if (sy >= src_height - 3) {
            fy = 0, sy = src_height - 3;
        }

        const double A = -0.75f;

        double coeffsY[4];
        coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
        coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
        coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
        coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

        for (int i = 0; i < dst_width; ++i) {
            double fx = (double) ((i + 0.5) * scale_x - 0.5);
            int sx = int(floor(fx));
            fx -= sx;

            if (sx < 1) {
                fx = 0, sx = 1;
            }
            if (sx >= src_width - 3) {
                fx = 0, sx = src_width - 3;
            }

            double coeffsX[4];
            coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
            coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
            coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
            coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

            for (int k = 0; k < channels; ++k) {
                dst_im[j * dstrows + i * channels + k] = (T) ((
                        src_im[(sy - 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[0] +
                        src_im[(sy) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[1] +
                        src_im[(sy + 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[2] +
                        src_im[(sy + 2) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[3] +

                        src_im[(sy - 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[0] +
                        src_im[(sy) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[1] +
                        src_im[(sy + 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[2] +
                        src_im[(sy + 2) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[3] +

                        src_im[(sy - 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[0] +
                        src_im[(sy) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[1] +
                        src_im[(sy + 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[2] +
                        src_im[(sy + 2) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[3] +

                        src_im[(sy - 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[0] +
                        src_im[(sy) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[1] +
                        src_im[(sy + 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[2] +
                        src_im[(sy + 2) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[3]));

            }//end k
        }
    }
}


#endif // _INC_RESIZE_BICUBIC_H
