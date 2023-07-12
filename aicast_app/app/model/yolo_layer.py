import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class YoloLayer():
    """
    Fast implementation of yolo layer.
    skip coordinate and confidence calculation for bboxes with confidence under threshold.
    
    apply_sigmoid: when model alls script contains 'change_output_activation(sigmoid)', set False
    """
    def __init__(
        self, 
        n_class,
        conf_thresh,
        apply_sigmoid=True,
        in_size=640, 
        output_sizes=[[80, 80], [40, 40], [20, 20]],
        anchors=[
            [10,13, 16,30, 33,23],
            [30,61, 62,45, 59,119],
            [116,90, 156,198, 373,326],
        ],
        n_max_output_bbox=30000
    ):
        self.n_class = n_class
        self.conf_thresh = conf_thresh
        self.activate_fn = sigmoid if apply_sigmoid else (lambda x: x)
        self.inv_conf_thres = np.log(conf_thresh/(1-conf_thresh))
        self.n_output = len(output_sizes)
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.n_output, self.n_output, 2)
        self.stride = [in_size // o[0] for o in output_sizes]
        self.grid = [
            self.make_grid(nx, ny)
            for nx, ny in output_sizes
        ]
        self.n_max_output_bbox = n_max_output_bbox

    def make_grid(self, nx, ny):
        y = np.arange(ny, dtype=np.float32)
        x = np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x, indexing='ij')
        return np.stack((xv, yv), 2) - 0.5

    def run(self, preds):
        """
        preds: list of model output np.ndarray
        len(preds) == len(self.output_sizes)
        By default, len(preds) == 3 and
        preds[0].shape = [80, 80, 3, (5 + self.n_class)]
        preds[1].shape = [40, 40, 3, (5 + self.n_class)]
        preds[2].shape = [20, 20, 3, (5 + self.n_class)]
        """
        boxes = []
        confs = []
        classes = []
        for i in range(self.n_output):
            # filter by object threshold
            iy, ix, ia = np.where(preds[i][..., 4] > self.inv_conf_thres)
            if len(iy) == 0:
                continue
            x = preds[i][iy, ix, ia, :]

            #compute bbox
            grid = self.grid[i][iy, ix]
            anchor_grid = self.anchors[i][ia]
            xy = (self.activate_fn(x[:, :2]) * 2 + grid) * self.stride[i]
            wh = ((self.activate_fn(x[:, 2:4]) * 2) ** 2) * anchor_grid
            cls = x[:, 5:5+self.n_class].argmax(1)
            class_conf = self.activate_fn(np.take_along_axis(x[:, 5:5+self.n_class], cls.reshape(-1, 1), axis=1)).flatten()
            obj_conf = self.activate_fn(x[:, 4])
            conf = class_conf * obj_conf
            boxes.append(np.hstack([xy-wh/2, xy+wh/2]))
            classes.append(cls)
            confs.append(conf)
        
        if len(boxes) == 0:
            return [], [], []

        boxes = np.vstack(boxes)
        classes = np.hstack(classes)
        confs = np.hstack(confs)

        # filter by conf
        sel = np.where(confs > self.conf_thresh)
        boxes = boxes[sel]
        classes = classes[sel]
        confs = confs[sel]
        # filter by bbox num
        sel = np.argsort(-confs, axis=0)[:self.n_max_output_bbox]
        boxes = boxes[sel]
        classes = classes[sel]
        confs = confs[sel]

        return boxes, confs, classes