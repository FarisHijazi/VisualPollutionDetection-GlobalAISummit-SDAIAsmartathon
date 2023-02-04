"""
gradio demo for yolov5
"""

import argparse
import random

import cv2
import gradio as gr
import torch


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        default='repositories/yolov5/smartathon/yolov5x-heavyaug/weights/best.pt',
        help='model.pt path',
    )
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', args.weights).eval()

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(model.names))]

    def run_detection(img):
        preds = model(img).pred
        if len(preds) > 0:
            pred = preds[0]
            for x1, y1, x2, y2, conf, cls in pred:
                label = model.names[int(cls)]
                plot_one_box([x1, y1, x2, y2], img, color=colors[int(cls)], label=label)
                # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.putText(img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

    # add slider for threshold

    # slider_input = gr.Slider(minimum=0.2,maximum=1,value=0.25,label='Prediction Threshold')
    css = "body {background-image: url('./bg.png'); repeat 0 0;}"
    app = gr.Interface(
        fn=run_detection,
        inputs=gr.Image(shape=(640, 640)),
        outputs=gr.Image(shape=(640, 640)),
        css=css,
    )
    app.launch(share=True, server_name='0.0.0.0')
