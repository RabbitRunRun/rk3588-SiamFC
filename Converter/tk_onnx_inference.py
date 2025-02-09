#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: tk_onnx_inference.py
# Copyright(c) 2017-2022 SeetaTech
# Written by Zhuang Liu
# 2024/05/27 17:54
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import onnxruntime
import numpy as np
import cv2


class Anchors:
    """
    This class generate anchors.
    """

    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(np.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1


def generate_anchor(score_size):
    anchors = Anchors(8,
                      [0.33, 0.5, 1, 2, 3],
                      [8])
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
    total_stride = anchors.stride
    anchor_num = anchor.shape[0]
    anchor = np.tile(anchor, score_size[0] * score_size[1]).reshape((-1, 4))
    orix = - (score_size[0] / 2.) * total_stride
    oriy = - (score_size[1] / 2.) * total_stride
    xx, yy = np.meshgrid([orix + total_stride * dx for dx in range(score_size[0])],
                         [oriy + total_stride * dy for dy in range(score_size[1])])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class Tracker:
    def __init__(self, fonnx_file_path, monnx_file_path, template_size, search_size):
        super(Tracker, self).__init__()
        ## 初始化ONNX推理会话
        # 加载导出的特征ONNX模型
        onnx_model = onnx.load(fonnx_file_path)
        # 验证模型
        onnx.checker.check_model(onnx_model)
        # 在ONNX Runtime中创建一个推理会话
        self.init_ort_session = onnxruntime.InferenceSession(fonnx_file_path)

        # 加载导出的匹配ONNX模型
        onnx_model = onnx.load(monnx_file_path)
        # 验证模型
        onnx.checker.check_model(onnx_model)
        # 在ONNX Runtime中创建一个推理会话
        self.track_ort_session = onnxruntime.InferenceSession(monnx_file_path)

        ##  初始化网络输入小图和大图的尺寸，匹配结果大小以及anchors
        self.template_size = template_size
        self.search_size = search_size
        self.output_size = ((search_size[0] - template_size[0]) // 8 + 1, (search_size[1] - template_size[1]) // 8 + 1)
        if template_size[0] < 160:
            self.output_size = (search_size[0] // 8 - 7 + 1, search_size[1] // 8 - 7 + 1)
        self.anchors = generate_anchor(self.output_size)

        # other track parameters
        self.anchor_num = 5
        self.PENALTY_K = 0.04
        self.WINDOW_INFLUENCE = 0.44
        hanning = np.hanning(self.output_size[0])
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.LR = 0.33

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        return im_patch

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scale_z = self.template_size[0] / s_z
        self.s_x = s_z * (self.search_size[0] / self.template_size[0])

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    self.template_size[0],
                                    s_z, self.channel_average)

        # 执行ONNX推理，获取模板特征
        inputs = {'img': z_crop}
        forward_outputs = self.init_ort_session.run(None, inputs)
        self.zf = forward_outputs

    def track(self, img):
        """
               args:
                   img(np.ndarray): BGR image
               return:
                   bbox(list):[x, y, width, height]
               """
        # 获取搜索图
        x_crop = self.get_subwindow(img, self.center_pos,
                                    self.search_size[0],
                                    round(self.s_x), self.channel_average)

        # 输入模版特征和搜索图，获取匹配结果
        inputs = {'template_feature1': self.zf[0],
                  'template_feature2': self.zf[1],
                  'template_feature3': self.zf[2],
                  'search_img': x_crop}
        outputs = self.track_ort_session.run(None, inputs)

        score = self._convert_score(outputs[0])
        pred_bbox = self._convert_bbox(outputs[1], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.WINDOW_INFLUENCE) + \
                 self.window * self.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / self.scale_z
        lr = penalty[best_idx] * score[best_idx] * self.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        return {
            'bbox': bbox,
            'best_score': best_score
        }

    def _convert_bbox(self, delta, anchor):
        delta = delta.transpose(1, 2, 3, 0).reshape(4, -1)

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def softmax(self, x):
        if len(x.shape) > 1:
            # 矩阵
            tmp = np.max(x, axis=1)
            x -= tmp.reshape((x.shape[0], 1))
            x = np.exp(x)
            tmp = np.sum(x, axis=1)
            x /= tmp.reshape((x.shape[0], 1))
        else:
            # 向量
            tmp = np.max(x)
            x -= tmp
            x = np.exp(x)
            tmp = np.sum(x)
            x /= tmp
        return x

    def _convert_score(self, score):
        score = score.transpose(1, 2, 3, 0).reshape(2, -1).transpose(1, 0)
        score = self.softmax(score)[:, 1]
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

class AlexnetTracker:
    def __init__(self, fonnx_file_path, monnx_file_path, template_size, search_size):
        super(AlexnetTracker, self).__init__()
        ## 初始化ONNX推理会话
        # 加载导出的特征ONNX模型
        onnx_model = onnx.load(fonnx_file_path)
        # 验证模型
        onnx.checker.check_model(onnx_model)
        # 在ONNX Runtime中创建一个推理会话
        self.init_ort_session = onnxruntime.InferenceSession(fonnx_file_path)

        # 加载导出的匹配ONNX模型
        onnx_model = onnx.load(monnx_file_path)
        # 验证模型
        onnx.checker.check_model(onnx_model)
        # 在ONNX Runtime中创建一个推理会话
        self.track_ort_session = onnxruntime.InferenceSession(monnx_file_path)

        ##  初始化网络输入小图和大图的尺寸，匹配结果大小以及anchors
        self.template_size = template_size
        self.search_size = search_size
        self.output_size = (17, 17)
        self.anchors = generate_anchor(self.output_size)

        # other track parameters
        self.anchor_num = 5
        self.PENALTY_K = 0.04
        self.WINDOW_INFLUENCE = 0.44
        hanning = np.hanning(self.output_size[0])
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.LR = 0.33

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        return im_patch

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scale_z = self.template_size[0] / s_z
        self.s_x = s_z * (self.search_size[0] / self.template_size[0])

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    self.template_size[0],
                                    s_z, self.channel_average)

        # 执行ONNX推理，获取模板特征
        inputs = {'img': z_crop}
        forward_outputs = self.init_ort_session.run(None, inputs)
        self.zf = forward_outputs

    def track(self, img):
        """
               args:
                   img(np.ndarray): BGR image
               return:
                   bbox(list):[x, y, width, height]
               """
        # 获取搜索图
        x_crop = self.get_subwindow(img, self.center_pos,
                                    self.search_size[0],
                                    round(self.s_x), self.channel_average)

        # 输入模版特征和搜索图，获取匹配结果
        inputs = {'template_feature': self.zf[0],
                  'search_img': x_crop}
        outputs = self.track_ort_session.run(None, inputs)

        score = self._convert_score(outputs[0])
        pred_bbox = self._convert_bbox(outputs[1], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.WINDOW_INFLUENCE) + \
                 self.window * self.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / self.scale_z
        lr = penalty[best_idx] * score[best_idx] * self.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        return {
            'bbox': bbox,
            'best_score': best_score
        }

    def _convert_bbox(self, delta, anchor):
        delta = delta.transpose(1, 2, 3, 0).reshape(4, -1)

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def softmax(self, x):
        if len(x.shape) > 1:
            # 矩阵
            tmp = np.max(x, axis=1)
            x -= tmp.reshape((x.shape[0], 1))
            x = np.exp(x)
            tmp = np.sum(x, axis=1)
            x /= tmp.reshape((x.shape[0], 1))
        else:
            # 向量
            tmp = np.max(x)
            x -= tmp
            x = np.exp(x)
            tmp = np.sum(x)
            x /= tmp
        return x

    def _convert_score(self, score):
        score = score.transpose(1, 2, 3, 0).reshape(2, -1).transpose(1, 0)
        score = self.softmax(score)[:, 1]
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height
    
class OneModelTracker:
    def __init__(self, file_path, template_size, search_size):
        super(OneModelTracker, self).__init__()
        ## 初始化ONNX推理会话
        # 加载导出的特征ONNX模型
        onnx_model = onnx.load(file_path)
        # 验证模型
        onnx.checker.check_model(onnx_model)
        # 在ONNX Runtime中创建一个推理会话
        self.init_ort_session = onnxruntime.InferenceSession(file_path)

        ##  初始化网络输入小图和大图的尺寸，匹配结果大小以及anchors
        self.template_size = template_size
        self.search_size = search_size
        self.output_size = ((search_size[0] - template_size[0]) // 8 + 1, (search_size[1] - template_size[1]) // 8 + 1)
        if template_size[0] < 160:
            self.output_size = (search_size[0] // 8 - 7 + 1, search_size[1] // 8 - 7 + 1)
        self.anchors = generate_anchor(self.output_size)

        # other track parameters
        self.anchor_num = 5
        self.PENALTY_K = 0.04
        self.WINDOW_INFLUENCE = 0.44
        hanning = np.hanning(self.output_size[0])
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.LR = 0.33

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        return im_patch

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scale_z = self.template_size[0] / s_z
        self.s_x = s_z * (self.search_size[0] / self.template_size[0])

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    self.template_size[0],
                                    s_z, self.channel_average)

        # 执行ONNX推理，获取模板特征
        self.template_input = z_crop

    def track(self, img):
        """
               args:
                   img(np.ndarray): BGR image
               return:
                   bbox(list):[x, y, width, height]
               """
        # 获取搜索图
        x_crop = self.get_subwindow(img, self.center_pos,
                                    self.search_size[0],
                                    round(self.s_x), self.channel_average)

        # 输入模版图和搜索图，获取匹配结果
        inputs = {'template_img': self.template_input,
                  'search_img': x_crop}
        outputs = self.init_ort_session.run(None, inputs)

        score = self._convert_score(outputs[0])
        pred_bbox = self._convert_bbox(outputs[1], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.WINDOW_INFLUENCE) + \
                 self.window * self.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / self.scale_z
        lr = penalty[best_idx] * score[best_idx] * self.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        return {
            'bbox': bbox,
            'best_score': best_score
        }

    def _convert_bbox(self, delta, anchor):
        delta = delta.transpose(1, 2, 3, 0).reshape(4, -1)

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def softmax(self, x):
        if len(x.shape) > 1:
            # 矩阵
            tmp = np.max(x, axis=1)
            x -= tmp.reshape((x.shape[0], 1))
            x = np.exp(x)
            tmp = np.sum(x, axis=1)
            x /= tmp.reshape((x.shape[0], 1))
        else:
            # 向量
            tmp = np.max(x)
            x -= tmp
            x = np.exp(x)
            tmp = np.sum(x)
            x /= tmp
        return x

    def _convert_score(self, score):
        score = score.transpose(1, 2, 3, 0).reshape(2, -1).transpose(1, 0)
        score = self.softmax(score)[:, 1]
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height
    
def tk_onnx_inference_video():
    fonnx_file_path = 'models/checkpoint_e150_f.onnx'
    monnx_file_path = 'models/checkpoint_e150_m.onnx'

    # 初始化跟踪器, 参数如下：
    # fonnx_file_path: 特征网络backbone+neck路径
    # monnx_file_path: 匹配网络路径
    # template_size: 小图输入大小
    # search_size: 大图输入大小
    # tracker = Tracker(fonnx_file_path, monnx_file_path, template_size=(127, 127), search_size=(255, 255))
    tracker = AlexnetTracker(fonnx_file_path, monnx_file_path, template_size=(127, 127), search_size=(255, 255))

    # 读取视频及设置捕获结果
    video_name = '20240627145630-02.mp4'
    cap = cv2.VideoCapture(video_name)
    first_frame = True
    while True:
        ret, frame = cap.read()
        if ret:
            if first_frame:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame)
                print(outputs['best_score'])
                bbox = list(map(int, outputs['bbox']))
                if outputs['best_score'] >= 0.3:
                    # cv2.rectangle(frame, (bbox[0], bbox[1]),
                    #             (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    #             (0, 255, 0), 3)
                    cv2.rectangle(frame, (100, -100),
                                (200, 200),
                                (0, 255, 0), 3)
                else:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                    tracker.init(frame, init_rect)
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)
        else:
            break


    
def tk_onnx_inference_video_one_model():
    onnx_file_path = 'models/checkpoint_e50_nodilation.onnx'

    # 初始化跟踪器, 参数如下：
    # fonnx_file_path: 特征网络backbone+neck路径
    # monnx_file_path: 匹配网络路径
    # template_size: 小图输入大小
    # search_size: 大图输入大小
    tracker = OneModelTracker(onnx_file_path, template_size=(127, 127), search_size=(255, 255))

    # 读取视频及设置捕获结果
    video_name = 'demo/20240426192512-01_cut.mp4'
    cap = cv2.VideoCapture(video_name)
    first_frame = True
    while True:
        ret, frame = cap.read()
        if ret:
            if first_frame:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame)
                print(outputs['best_score'])
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 3)
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)
        else:
            break

def tk_onnx_inference_images():
    fonnx_file_path = 'models/checkpoint_e50_f.onnx'
    monnx_file_path = 'models/checkpoint_e50_m.onnx'

    # 初始化跟踪器, 参数如下：
    # fonnx_file_path: 特征网络backbone+neck路径
    # monnx_file_path: 匹配网络路径
    # template_size: 小图输入大小
    # search_size: 大图输入大小
    tracker = Tracker(fonnx_file_path, monnx_file_path, template_size=(127, 127), search_size=(255, 255))

    # 读取视频及设置捕获结果
    template_image_file = 'images_origin/1060.png'
    to_search_image_file = 'images_origin/1065.png'
    template_frame = cv2.imread(template_image_file)
    search_frame = cv2.imread(to_search_image_file)
    bbox=[215.83545, 272.5504, 59.3732, 30.94336]
    tracker.init(template_frame, bbox=bbox)

    outputs = tracker.track(search_frame)
    print(outputs['best_score'])
    result_bbox = list(map(int, outputs['bbox']))
    bbox = list(map(int, bbox))

    # green is the target
    cv2.rectangle(search_frame, (result_bbox[0], result_bbox[1]),
                    (result_bbox[0] + result_bbox[2], result_bbox[1] + result_bbox[3]),
                    (0, 255, 0), 3)
    
    # red is origin positon
    cv2.rectangle(search_frame, (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (255, 0, 0), 3)
    cv2.imshow("image", search_frame)
    cv2.imwrite('result.png', search_frame)
    cv2.waitKey(10000)

if __name__ == '__main__':
    # tk_onnx_inference_images()
    # tk_onnx_inference_video_one_model()
    tk_onnx_inference_video()
