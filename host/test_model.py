import argparse
import cv2
import numpy as np
from hailo_sdk_client import ClientRunner
from hailo_sdk_common.targets.inference_targets import SdkNative, SdkFPOptimized, SdkPartialNumeric, SdkNumeric
import tensorflow as tf
from util.yolo_layer import YoloLayer
from util.nms import nms
from util.bbox_drawer import Drawer

def get_input():
    img = cv2.imread("/workspaces/handson/yolov5/data/images/bus.jpg")
    resized_img = cv2.resize(img, (640, 640))
    input = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    input = input.astype(dtype=np.float32)
    input = input[None, :, :, :]
    return resized_img, input

def net_eval(runner, target, images):
    with tf.Graph().as_default():
        network_input = tf.compat.v1.placeholder(dtype=tf.float32)
        sdk_export = runner.get_tf_graph(target, network_input)
        with sdk_export.session.as_default():
            sdk_export.session.run(tf.compat.v1.local_variables_initializer())
            probs_batch = sdk_export.session.run(sdk_export.output_tensors, feed_dict={network_input: images})
    return probs_batch

def postprocess(resized_img, out0, out1, out2):
    out0 = out0.reshape(80, 80, 3, 85)
    out1 = out1.reshape(40, 40, 3, 85)
    out2 = out2.reshape(20, 20, 3, 85)
    yolox_layer = YoloLayer(80, 0.5, in_size=640)
    boxes, confs, classes = yolox_layer.run([out0, out1, out2])
    boxes, confs, classes = nms(boxes, confs, classes)
    drawer = Drawer()
    result_img = drawer.draw(resized_img, boxes, confs, classes)
    return result_img

    
def test_quantized():
    quantized_model = "../make_hef/yolov5s_quantized.har"
    runner = ClientRunner(hw_arch='hailo8r', har_path=quantized_model)
    resized_img, input = get_input()
    out0, out1, out2 = net_eval(runner, SdkNumeric(), input)
    result_img = postprocess(resized_img, out0, out1, out2)
    cv2.imwrite("result_quantized.jpg", result_img)

def test_float():
    model = "../make_hef/yolov5s.har"
    runner = ClientRunner(hw_arch='hailo8r', har_path=model)
    resized_img, input = get_input()
    input = input / 255.
    out0, out1, out2 = net_eval(runner, SdkNative(), input)
    result_img = postprocess(resized_img, out0, out1, out2)
    cv2.imwrite("result_float.jpg", result_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str)
    args = parser.parse_args()
    if args.target == "float":
        test_float()
    elif args.target == "quantized":
        test_quantized()