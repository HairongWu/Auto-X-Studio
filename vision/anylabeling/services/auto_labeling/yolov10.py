import logging
import os

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
import ultralytics

class YOLOv10(Model):
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
            "nms_threshold",
            "confidence_threshold",
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

    def post_process(self, input_image, outputs):
        """
        Post-process the network's output, to get the bounding boxes and
        their confidence scores.
        """
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []

        output_boxes = []
        for i in outputs:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            label = self.classes[class_ids[i]]
            score = confidences[i]

            output_box = {
                "x1": left,
                "y1": top,
                "x2": left + width,
                "y2": top + height,
                "label": label,
                "score": score,
            }

            output_boxes.append(output_box)

        return output_boxes

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []
        
        detections = self.net(image)
        boxes = self.post_process(image, detections)
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
