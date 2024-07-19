import os
import numpy as np
import cv2
from rknn.api import RKNN
import time
import glob
from logdown import logdown

QUANTIZE_ON = True
QUANT_IMG_RGB2BGR = True
_force_builtin_perm = False


def convert_f_model():
    DATASET = './dataset_f_model.txt'
    ONNX_MODEL = 'models/checkpoint_e150_f.onnx'
    RKNN_MODEL = 'rknn_models/checkpoint_e150_f_u8.rknn'
    # Create RKNN object
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)


    # pre-process config
    print('--> Config model')
    rknn.config(target_platform = 'rk3588', quant_img_RGB2BGR=QUANT_IMG_RGB2BGR)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    image_file = 'test_image/1020_template.png'
    image = cv2.imread(image_file)
    # hwc->chw
    # image = image.transpose([2,0,1])
    outputs = rknn.inference(inputs=[image], inputs_pass_through=[0 if not _force_builtin_perm else 1])
    # print(type(outputs))
    # print("outputs len:", len(outputs))
    # for output in outputs:
    #     print(output)
    #     print('output shape is ', output.shape)

    # save template features in nhwc format,do not forget to convert format from nchw -> nhwc
    # np.save('test_image_results/template_feature1.npy', outputs[0].transpose([0,2,3,1]))
    # np.save('test_image_results/template_feature2.npy', outputs[1].transpose([0,2,3,1]))
    # np.save('test_image_results/template_feature3.npy', outputs[2].transpose([0,2,3,1]))
    np.save('test_image_results/template_feature.npy', outputs[0].transpose([0,2,3,1]))

    rknn.release()


def convert_m_model():
    DATASET = './dataset_m_model.txt'
    ONNX_MODEL = 'models/checkpoint_e150_m.onnx'
    RKNN_MODEL = 'rknn_models/checkpoint_e150_m_u8.rknn'
    # Create RKNN object
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)


    # pre-process config
    print('--> Config model')
    rknn.config(target_platform = 'rk3588', quant_img_RGB2BGR=QUANT_IMG_RGB2BGR)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference

    # load features input from f model outputs
    print('--> Running model')
    # feature1 = np.load('test_image_results/template_feature1.npy')
    # print('template feature1 shape:', feature1.shape)
    # feature2 = np.load('test_image_results/template_feature2.npy')
    # print('template feature2 shape:', feature2.shape)
    # feature3 = np.load('test_image_results/template_feature3.npy')
    # print('template feature3 shape:', feature3.shape)
    feature = np.load('test_image_results/template_feature.npy')

    # which image to search
    image_file = 'to_quantization_images/alexnet/m_model/t1020_s1021/search.png'
    image = cv2.imread(image_file)
    print('search image shape: ', image.shape)
    outputs = rknn.inference(inputs=[feature, image], inputs_pass_through=None) # inputs_pass_through=None默认不透传
    print("outputs len:", len(outputs))
    # for output in outputs:
    #     print(output)
    #     print('output shape is ', output.shape)

    # save outputs to npy file in nchw format, no need to convert format
    np.save('test_image_results/score.npy', outputs[0])
    np.save('test_image_results/pos.npy', outputs[1])

    rknn.release()


def convert_model():
    DATASET = './dataset_f_model.txt'
    ONNX_MODEL = 'models/checkpoint_e50.onnx'
    RKNN_MODEL = 'rknn_models/checkpoint_e50_fp16.rknn'
    # Create RKNN object
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)


    # pre-process config
    print('--> Config model')
    rknn.config(target_platform = 'rk3588', quant_img_RGB2BGR=QUANT_IMG_RGB2BGR)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(perf_debug=True)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    image_file = 'test_image/1060_template.png'
    image_t = cv2.imread(image_file)
    # which image to search
    image_file = 'to_quantization_images/m_model/t1060_s1061/search.png'
    image_s = cv2.imread(image_file)
    print('search image shape: ', image_s.shape)
    # hwc->chw
    # image = image.transpose([2,0,1])

    #accuracy analysis
    # ret = rknn.accuracy_analysis(inputs=[image_t, image_s])


    outputs = rknn.inference(inputs=[image_t, image_s], inputs_pass_through=[0, 0])
    print(type(outputs))
    print("outputs len:", len(outputs))
    # for output in outputs:
    #     print(output)
    #     print('output shape is ', output.shape)

    # save outputs to npy file in nchw format, no need to convert format
    np.save('test_image_results/total_score.npy', outputs[0])
    np.save('test_image_results/total_pos.npy', outputs[1])

    rknn.release()



def convert_model_with_abundan_outputs():
    DATASET = './dataset_f_model.txt'
    ONNX_MODEL = 'models/your_model_with_new_output.onnx'
    RKNN_MODEL = 'rknn_models/your_model_with_new_output_fp16.rknn'
    # Create RKNN object
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)


    # pre-process config
    print('--> Config model')
    rknn.config(target_platform = 'rk3588', quant_img_RGB2BGR=QUANT_IMG_RGB2BGR)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(perf_debug=True)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    image_file = 'test_image/1060_template.png'
    image_t = cv2.imread(image_file)
    # which image to search
    image_file = 'to_quantization_images/m_model/t1060_s1061/search.png'
    image_s = cv2.imread(image_file)
    # image_t = cv2.resize(image_t, (128, 128))
    # image_s = cv2.resize(image_s, (256,256))
    # cv2.imwrite('128x128.png', image_t)
    # cv2.imwrite('256x256.png', image_s)
    print('search image shape: ', image_s.shape)
    # hwc->chw
    # image = image.transpose([2,0,1])

    #accuracy analysis
    # ret = rknn.accuracy_analysis(inputs=[image_t, image_s])


    outputs = rknn.inference(inputs=[image_t, image_s], inputs_pass_through=[0, 0])
    print(type(outputs))
    print("outputs len:", len(outputs))
    # for output in outputs:
    #     print(output)
    #     print('output shape is ', output.shape)

    # save outputs
    for idx in range(len(outputs)):
        np.savetxt(os.path.join("debug", "output" + str(idx) + ".txt"), outputs[idx].flatten())

    rknn.release()


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
    def __init__(self, template_size, search_size):
        super(Tracker, self).__init__()

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

    def init(self, bbox):
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

    def track(self, img):
       
        output_score = np.load('test_image_results/score.npy')
        output_pos = np.load('test_image_results/pos.npy')
        print('output score shape in tracker:', output_score.shape)
        print('output pos shape in tracker:', output_pos.shape)
        print("output_pos:", output_pos)
        print("output_score:", output_score)
        outputs = [output_score, output_pos]
        score = self._convert_score(outputs[0])
        # print('score:', score)
        pred_bbox = self._convert_bbox(outputs[1], self.anchors)
        print("###after self._convert_bbox pred_bbox:", pred_bbox)
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        print("###self.size:", self.size)
        print("###self.anchors:", self.anchors)
        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z)))
        print("###s_c:", s_c)
        print("###self.scale_z:", self.scale_z)

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        print("###r_c:", r_c)
        penalty = np.exp(-(r_c * s_c - 1) * self.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.WINDOW_INFLUENCE) + \
                 self.window * self.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        print("###self.window:", self.window)

        bbox = pred_bbox[:, best_idx] / self.scale_z
        print("###bbox:", bbox)
        lr = penalty[best_idx] * score[best_idx] * self.LR
        print("###penalty:", penalty)
        print("###best_idx:", best_idx)
        print("###lr:", lr)
        print("###self.center_pos", self.center_pos)
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
    def __init__(self, template_size, search_size):
        super(AlexnetTracker, self).__init__()

        self.template_size = template_size
        self.search_size = search_size
        self.output_size = ((search_size[0] - template_size[0]) // 8 + 1, (search_size[1] - template_size[1]) // 8 + 1)
        if template_size[0] < 160:
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

    def init(self, bbox):
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

    def track(self, img):
       
        output_score = np.load('test_image_results/score.npy')
        output_pos = np.load('test_image_results/pos.npy')
        # print('output score shape in tracker:', output_score.shape)
        # print('output pos shape in tracker:', output_pos.shape)
        # print("output_pos:", output_pos)
        # print("output_score:", output_score)
        outputs = [output_score, output_pos]
        score = self._convert_score(outputs[0])
        # print('score:', score)
        pred_bbox = self._convert_bbox(outputs[1], self.anchors)
        # print("###after self._convert_bbox pred_bbox:", pred_bbox)
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # print("###self.size:", self.size)
        # print("###self.anchors:", self.anchors)
        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z)))
        # print("###s_c:", s_c)
        # print("###self.scale_z:", self.scale_z)

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        # print("###r_c:", r_c)
        penalty = np.exp(-(r_c * s_c - 1) * self.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.WINDOW_INFLUENCE) + \
                 self.window * self.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        # print("###self.window:", self.window)

        bbox = pred_bbox[:, best_idx] / self.scale_z
        # print("###bbox:", bbox)
        lr = penalty[best_idx] * score[best_idx] * self.LR
        # print("###penalty:", penalty)
        # print("###best_idx:", best_idx)
        # print("###lr:", lr)
        # print("###self.center_pos", self.center_pos)
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
    
def get_tracker_simulator_results():
    tracker = AlexnetTracker(template_size=(127, 127), search_size=(255, 255))

    #  1060.png target pos is [215.83545, 272.5504, 59.3732, 30.94336] in [x, y, w, h] format
    # 1020.png target pos is [333, 238, 49, 31]
    tracker.init(bbox=[333, 238, 49, 31])

    # tracked image is 1065.png
    tracked_image = cv2.imread('test_image/1021.png')
    outputs = tracker.track(tracked_image)
    print("best_score:", outputs['best_score'])
    bbox = list(map(int, outputs['bbox']))
    cv2.rectangle(tracked_image, (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (0, 255, 0), 3)
    print("[x,y,width,height]=[{},{},{},{}]:".format(bbox[0], bbox[1], bbox[2], bbox[3]))
    cv2.imwrite("test_image_results/result.png", tracked_image)

def read_score_txt_and_post():
    
    def softmax(x):
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

    def _convert_score(score):
        score = score.transpose(1, 2, 3, 0).reshape(2, -1).transpose(1, 0)
        score = softmax(score)[:, 1]
        return score
    
    file = 'test.txt'
    with open(file, 'r', encoding='utf-8') as reader:
        line = reader.readline()
        line = line.strip(' ').rstrip(' ')
        numbers = [float(number) for number in line.split(' ') if number != ' ']
        print(numbers)
        scores = np.asarray(numbers).reshape(1, 10, 25, 25)
        converted_score = _convert_score(scores.copy())
        print("converted score max:", np.max(converted_score))
        index = np.argmax(converted_score)
        print('max index:', index)
        print("max score pair:", scores.reshape(2, -1)[:, index])


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def rknn_debug_with_simulator_debug_analysis():
    rknn_debug_txts = glob.glob('rknn_debug/*.txt')
    debug_txts = glob.glob('debug/*.txt')
    # print(rknn_debug_txts)
    # print(debug_txts)
    # print(len(rknn_debug_txts))
    # print(len(debug_txts))
    assert len(rknn_debug_txts) == len(debug_txts)
    for idx in range(len(rknn_debug_txts)):
        debug_data = np.loadtxt(f'debug/output{idx}.txt').flatten()
        # print('debug_data shape:', debug_data.shape)
        rknn_debug_data = np.loadtxt(f'rknn_debug/output{idx}.txt').flatten()
        # print('rknn_debug_dat shape:', rknn_debug_data.shape)
        assert debug_data.shape == rknn_debug_data.shape
        difference = debug_data - rknn_debug_data
        # print(f'difference{idx} max:', np.max(difference))
        # print(f'difference{idx} min:', np.min(difference))
        print(f'difference{idx} mean:', np.mean(difference))
        print(f'difference{idx} var:', np.var(difference))
        print(f'cosine sim:', cosine_similarity(debug_data, rknn_debug_data))


if __name__ == '__main__':
    with logdown():
        convert_f_model()
        convert_m_model()
        get_tracker_simulator_results()

    # convert_model()
    # rknn = RKNN()
    # print(rknn.list_devices())
    # rknn.release()

    # convert_model_with_abundan_outputs()
    # rknn_debug_with_simulator_debug_analysis()

    # get_tracker_simulator_results()

    # f1 = np.load("./test_image_results/template_feature1.npy")
    # f2 = np.load("./test_image_results/template_feature2.npy")
    # f3 = np.load("./test_image_results/template_feature3.npy")
    # score = np.load("./test_image_results/total_score.npy")
    # pos = np.load("./test_image_results/total_pos.npy")
    # np.savetxt('test_image_results/total_score.txt', score.flatten(), fmt="%f", delimiter=" ")
    # np.savetxt('test_image_results/total_pos.txt', pos.flatten(), fmt="%f", delimiter=" ")
    # print("f1 shape:", f1.shape)
    # print("f1:", f1[0,0,0, :10])
    # print("f2:", f2[0,0,0,:10])
    # print("f3:", f3[0,0,0,:10])
    # print("f1 mean:", f1.mean())
    # print("f2 mean:", f2.mean())
    # print("f3 mean:", f3.mean())
    # print("score mean:", score.mean())
    # print("pos mean:", pos.mean())
    # print("f1 var:", np.var(f1))
    # print("f2 var:", np.var(f2))
    # print("f3 var:", np.var(f3))
    # print("score var:", np.var(score))
    # print("pos var:", np.var(pos))

    # score = score.reshape(2,-1)
    # print("total_score 1 col:", score[1,:])
    # print("total_score max:", np.max(score[1, :]))
    # index = np.argmax(score[1, :])
    # print('max index:', index)
    # print("max score pair:", score[:, index])

    # read_score_txt_and_post()


