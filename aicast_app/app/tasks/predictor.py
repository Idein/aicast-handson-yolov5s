import consts
from actfw_core.task import Pipe
from model import YOLOv5s
from PIL import Image


class Predictor(Pipe):
    def __init__(self, settings, capture_size):
        super(Predictor, self).__init__()
        self.settings = settings
        self.capture_size = capture_size
        self.model = YOLOv5s(n_class=80, conf_thresh=settings["thresh"], input_img_size=(
            consts.CAPTURE_WIDTH, consts.CAPTURE_HEIGHT))

    def get_capture_image(self, frame) -> Image:
        captured_image = Image.frombuffer(
            'RGB', self.capture_size, frame.getvalue(), 'raw', 'RGB')
        if self.capture_size != (consts.CAPTURE_WIDTH, consts.CAPTURE_HEIGHT):
            w, h = self.capture_size
            captured_image = captured_image.crop(
                (w // 2 - consts.CAPTURE_WIDTH // 2, h // 2 - consts.CAPTURE_HEIGHT // 2, w // 2 + consts.CAPTURE_WIDTH // 2, h // 2 + consts.CAPTURE_HEIGHT // 2))
        return captured_image

    def proc(self, frame):
        captured_image = self.get_capture_image(frame)
        bboxes, scores, classes = self.model.run(captured_image)
        return captured_image, bboxes, scores, classes
