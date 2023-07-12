from ctypes import cdll

import numpy as np


class Model:
    def __init__(self):
        self.lib = cdll.LoadLibrary('./libhailomodels.so')
        self.lib.init()
        # 7 = xywh (4) + obj_conf (1) + cls_conf (2)
        self.pred = [
            np.zeros((80, 80, 3, 7), dtype=np.float32),
            np.zeros((40, 40, 3, 7), dtype=np.float32),
            np.zeros((20, 20, 3, 7), dtype=np.float32)
        ]

    def __del__(self):
        self.lib.destroy()

    def infer(self, image):
        self.lib.infer(
            image.ctypes.data,
            self.pred[0].ctypes.data,
            self.pred[1].ctypes.data,
            self.pred[2].ctypes.data,
        )

        return self.pred
