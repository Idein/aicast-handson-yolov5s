# from ctypes import cdll
import numpy as np
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou(a, b):
    ax_min, ay_min, ax_max, ay_max = a
    bx_min, by_min, bx_max, by_max = b

    l = max(ax_min, bx_min)
    r = min(ax_max, bx_max)
    u = max(ay_min, by_min)
    d = min(ay_max, by_max)

    w = max(r - l, 0)
    h = max(d - u, 0)
    aw = ax_max - ax_min
    ah = ay_max - ay_min
    bw = bx_max - bx_min
    bh = by_max - by_min

    return w * h / (aw * ah + bw * bh - w * h)

class Model:
    def __init__(self, conf_thres=0.25, iou_thres=0.45, max_wh=7680, max_nms=30000):
        self.inv_conf_thres = np.log(conf_thres/(1-conf_thres))
        self.iou_thres = iou_thres
        self.max_wh = max_wh
        self.max_nms = max_nms
        self.insize = 640
        self.nclass = 2

        self.anchors = np.array([
            [10,13, 16,30, 33,23],
            [30,61, 62,45, 59,119],
            [116,90, 156,198, 373,326],
            ], dtype=np.float32).reshape(3, 3, 2)
        self.stride = [
            self.insize // 80,
            self.insize // 40,
            self.insize // 20
            ]
        self.grid = [
            self.make_grid(nx, ny)
            for nx, ny in [(80, 80), (40, 40), (20, 20)]
            ]

    def __del__(self):
        pass

    def make_grid(self, nx, ny):
        y = np.arange(ny, dtype=np.float32)
        x = np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x, indexing='ij')
        return np.stack((xv, yv), 2) - 0.5

    def infer(self, intermediate_inputs):
        # (1, 24, 80/40/20, 80/40/20) -> (80/40/20, 80/40/20, 21)
        intermediate_inputs = [np.transpose(x[0], (0, 1, 2)) for x in intermediate_inputs]
        # (80/40/20, 80/40/20) -> (80/40/20, 80/40/20, 3, 7)
        intermediate_inputs = [x.reshape(s, s, 3, 7) for (x, s) in zip(intermediate_inputs, [80, 40, 20])]

        start = time.time()
        boxes = []
        confs = []
        classes = []

        for i in range(3):
            # note:
            #   sigmoid(x) > t
            #   <-> x > ln(t/(1-t))
            # iy, ix, ia = np.where(self.pred[i][..., 4] > self.inv_conf_thres)
            iy, ix, ia = np.where(intermediate_inputs[i][..., 4] > self.inv_conf_thres)
            if len(iy) == 0:
                continue
            x = intermediate_inputs[i][iy, ix, ia, :]

            # compute bbox
            grid = self.grid[i][iy, ix]
            anchor_grid = self.anchors[i][ia]
            # x: (n, 117)
            # grid: (n, 2)
            # anchor_grid: (n, 2)

            xy = (sigmoid(x[:, :2]) * 2 + grid) * self.stride[i]
            wh = ((sigmoid(x[:, 2:4]) * 2) ** 2) * anchor_grid
            cls = x[:, 5:5+self.nclass].argmax(1)
            conf = sigmoid(np.take_along_axis(x[:, 5:5+self.nclass], cls.reshape(-1, 1), axis=1))

            
            boxes.append(np.hstack([xy-wh/2, xy+wh/2]))
            classes.append(cls)
            confs.append(conf)
        if len(boxes) == 0:
            return [], [], []

        boxes = np.vstack(boxes)
        classes = np.hstack(classes)
        confs = np.vstack(confs).flatten()

        sel = np.argsort(-confs, axis=0)[:self.max_nms]
        boxes = boxes[sel]
        classes = classes[sel]
        confs = confs[sel]

        # non-maximum suppression
        nmsed_candidates = []
        candidates = list(range(len(boxes)))
        while len(candidates) > 0:
            a = candidates.pop(0)
            nmsed_candidates.append(a)
            a_box = boxes[a]
            def nms(b):
                b_box = boxes[b]
                return iou(a_box, b_box) <= self.iou_thres
            candidates = list(filter(nms, candidates))
        boxes = boxes[nmsed_candidates]
        classes = classes[nmsed_candidates]
        confs = confs[nmsed_candidates]
        print(len(boxes), 1/(time.time()-start), flush=True)
        return boxes, classes, confs