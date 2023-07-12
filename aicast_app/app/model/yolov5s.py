from ctypes import cdll
from typing import List, Tuple

import numpy as np
from PIL import Image

from .nms import nms
from .yolo_layer import YoloLayer


class LetterBoxDecoder():
    def __init__(self, input_size, letterbox_size=(640, 640)):
        self.letterbox_size = letterbox_size
        width, height = input_size
        lw, lh = letterbox_size
        self.ratio = min(lw / width, lw / height)
        self.new_width = int(width * self.ratio)
        self.new_height = int(height * self.ratio)
        self.x_offset = int((lw - self.new_width) / 2)
        self.y_offset = int((lh - self.new_height) / 2)
        self.bg_color = (112, 112, 112)

    def get_letterbox_image(self, pil_img: Image):
        resized_image = pil_img.resize((self.new_width, self.new_height))
        letterbox_image = Image.new('RGB', self.letterbox_size, self.bg_color)
        letterbox_image.paste(resized_image, (self.x_offset, self.y_offset))
        return letterbox_image

    def decode_box_letter_to_orig(self, boxes):
        """decode box coordinate from letterbox to input image"""
        if len(boxes) == 0:
            return np.empty((0, 4))
        return np.array([[
            (box[0] - self.x_offset) / self.ratio,
            (box[1] - self.y_offset) / self.ratio,
            (box[2] - self.x_offset) / self.ratio,
            (box[3] - self.y_offset) / self.ratio]
            for box in boxes])


class YOLOv5s:
    def __init__(self, n_class: int, conf_thresh: float, input_img_size=Tuple[int, int]):
        self.lib = cdll.LoadLibrary('./libhailomodels.so')
        self.lib.init()
        self.out0 = np.zeros((80, 80, 3, n_class + 5), dtype=np.float32)
        self.out1 = np.zeros((40, 40, 3, n_class + 5), dtype=np.float32)
        self.out2 = np.zeros((20, 20, 3, n_class + 5), dtype=np.float32)
        self.yolo_layer = YoloLayer(
            n_class=n_class, in_size=640, conf_thresh=conf_thresh, apply_sigmoid=True)
        self.lb_decoder = LetterBoxDecoder(input_img_size, (640, 640))

    def __del__(self):
        self.lib.destroy()

    def preproc(self, image: Image) -> np.ndarray:
        lb_img = self.lb_decoder.get_letterbox_image(image)
        return np.asarray(lb_img)

    def infer(self, input: np.ndarray):
        self.lib.infer(
            input.ctypes.data,
            self.out0.ctypes.data,
            self.out1.ctypes.data,
            self.out2.ctypes.data)

        return self.out0, self.out1, self.out2

    def postprocess(self, outs: List[np.ndarray]):
        bboxes, scores, classes = self.yolo_layer.run(outs)
        bboxes, scores, classes = nms(
            bboxes, scores, classes, per_class=True, iou_threshold=0.45)
        bboxes = self.lb_decoder.decode_box_letter_to_orig(bboxes)
        return bboxes, scores, classes

    def run(self, image: Image):
        input = self.preproc(image)
        outs = self.infer(input)
        bboxes, scores, classes = self.postprocess(outs)
        return bboxes, scores, classes
