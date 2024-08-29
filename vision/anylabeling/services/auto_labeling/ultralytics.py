import logging
import os

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from .model import Model
from .types import AutoLabelingResult
import ultralytics

class Detection(Model):
    """Object detection model using YOLOv8"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "input_width",
            "input_height",
            "score_threshold",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8 model."
                )
            )
        self.net = ultralytics.YOLOv10(model_abs_path)
        self.classes = self.config["classes"]

    def pre_process(self, input_image, net):
        """
        Pre-process the input image before feeding it to the network.
        """
        # Create a 4D blob from a frame.
        pass

    def post_process(self, outputs):
        """
        Post-process the network's output, to get the bounding boxes and
        their confidence scores.
        """
        # Lists to hold respective values while unwrapping.

        output_boxes = []
        
        for result in outputs:
            boxes = result.boxes
            names = result.names
            
            for cls, conf, box in zip(boxes.cls, boxes.conf, boxes.xyxy):
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                output_box = {
                    "x1": left,
                    "y1": top,
                    "x2": left + width,
                    "y2": top + height,
                    "label": names[int(cls)],
                    "score": conf,
                }

                output_boxes.append(output_box)

        return output_boxes

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        detections = self.net(image_path)
        
        boxes = self.post_process(detections)
        shapes = []

        for box in boxes:
            shape = Shape(label=box["label"], shape_type="rectangle", flags={})
            shape.add_point(QtCore.QPointF(box["x1"], box["y1"]))
            shape.add_point(QtCore.QPointF(box["x2"], box["y2"]))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net

class Keypoint(Model):
    """Object detection model using YOLOv8"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "input_width",
            "input_height",
            "score_threshold",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8 model."
                )
            )
        self.net = ultralytics.YOLO(model_abs_path)
        self.classes = self.config["classes"]

    def pre_process(self, input_image, net):
        """
        Pre-process the input image before feeding it to the network.
        """
        # Create a 4D blob from a frame.
        pass

    def post_process(self, outputs):
        """
        Post-process the network's output, to get the bounding boxes and
        their confidence scores.
        """
        # Lists to hold respective values while unwrapping.

        output_boxes = []
        output_kps = []
        
        for result in outputs:
            print(result)
            boxes = result.boxes
            keypoints = result.keypoints
            names = result.names
            
            for cls, conf, box in zip(boxes.cls, boxes.conf, boxes.xyxy):
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                output_box = {
                    "x1": left,
                    "y1": top,
                    "x2": left + width,
                    "y2": top + height,
                    "label": names[int(cls)],
                    "score": conf,
                }

                output_boxes.append(output_box)
            
            for points in keypoints.xy:
                for i, point in enumerate(points):
                    x = point[0]
                    y = point[1]
                    output_kp = {
                        "x": x,
                        "y": y,
                        "label": str(i),
                    }

                    output_kps.append(output_kp)

        return output_boxes, output_kps

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        detections = self.net(image_path)
        
        boxes, keypoints = self.post_process(detections)
        shapes = []

        for box in boxes:
            shape = Shape(label=box["label"], shape_type="rectangle", flags={})
            shape.add_point(QtCore.QPointF(box["x1"], box["y1"]))
            shape.add_point(QtCore.QPointF(box["x2"], box["y2"]))
            shapes.append(shape)
        
        for kp in keypoints:
            shape = Shape(label=kp["label"], shape_type="point", flags={})
            shape.add_point(QtCore.QPointF(kp["x"], kp["y"]))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
