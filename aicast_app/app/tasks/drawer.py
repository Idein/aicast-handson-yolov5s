import time

import consts
from actfw_core.task import Pipe
from PIL import ImageDraw, ImageFont


class FPS(object):
    def __init__(self, moving_average=30):
        self.moving_average = moving_average
        self.prev_time = time.time()
        self.dtimes = []

    def update(self):
        cur_time = time.time()
        dtime = cur_time - self.prev_time
        self.prev_time = cur_time
        self.dtimes.append(dtime)
        if len(self.dtimes) > self.moving_average:
            self.dtimes.pop(0)
        return self.get()

    def get(self):
        if len(self.dtimes) == 0:
            return None
        else:
            return len(self.dtimes) / sum(self.dtimes)


class Drawer(Pipe):
    def __init__(self):
        super(Drawer, self).__init__()
        self.font = ImageFont.truetype(
            font='/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', size=14)
        self.fps = FPS(30)

    def proc(self, inputs):
        captured_image, bboxes, scores, classes = inputs

        self.fps.update()
        drawer = ImageDraw.Draw(captured_image)
        fps = self.fps.get()
        if fps is None:
            fps_txt = 'FPS: N/A'
        else:
            fps_txt = 'FPS: {:>6.3f}'.format(fps)
        drawer.text((0, 0), fps_txt, font=self.font, fill='white')
        cols = [(255, 0, 0), (0, 255, 0)]
        for box, score, cls in zip(bboxes, scores, classes):
            x1, y1, x2, y2 = [int(b) for b in box]
            drawer.rectangle((x1, y1, x2, y2), fill=None,
                             outline=cols[0], width=2)
            text = consts.coco_labels[cls] + ":{:.2f}%".format(score * 100)
            text_width, text_height = drawer.textsize(text, font=self.font)
            text_position = (x1, y1 - text_height)
            drawer.rectangle((x1, y1 - text_height, x1 +
                              text_width, y1), fill='red')
            drawer.text(text_position, text, fill='white', font=self.font)

        return captured_image
