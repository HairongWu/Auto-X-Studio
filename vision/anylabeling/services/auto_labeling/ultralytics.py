import logging
import os
import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from .model import Model
from .types import AutoLabelingResult

import onnxruntime
import threading
import numpy
from collections import namedtuple
from typing import Any, List

Session = onnxruntime.InferenceSession
Image = numpy.ndarray[Any, Any]
Prediction = numpy.ndarray[Any, Any]
MetaData = dict[str, list[int]]
Bbox = numpy.ndarray[Any, Any]
Kps = numpy.ndarray[Any, Any]
Score = float
Face = namedtuple('Face',
[
	'bbox',
	'kps',
	'score',
])

score_threshold = 0.25
iou_threshold = 0.4

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
        self.classes = self.config["classes"]
        with threading.Lock():
            self.model = onnxruntime.InferenceSession(model_abs_path)

    def preprocess_images(self, images: List[Image], input_size=(640, 640)) -> tuple[numpy.ndarray[Any, Any], MetaData]:
        preprocessed = []
        meta_data = {'ratios': [], 'dws': [], 'dhs': []}
        for image in images:
            # resize
            shape = image.shape[:2]
            ratio = min(input_size[0] / shape[0], input_size[1] / shape[1])
            new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
            dw, dh = (input_size[1] - new_unpad[0]) / 2, (input_size[0] - new_unpad[1]) / 2
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = round(dh - 0.1), round(dh + 0.1)
            left, right = round(dw - 0.1), round(dw + 0.1)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
            # normalize
            image = image.astype(numpy.float32) / 255.0
            
            # RGB -> BGR
            image = image[..., ::-1]
            
            # HWC -> CHW
            image = image.transpose((2, 0, 1))
            
            preprocessed.append(image)
            meta_data['ratios'].append(ratio)
            meta_data['dws'].append(dw)
            meta_data['dhs'].append(dh)
        return numpy.ascontiguousarray(numpy.array(preprocessed)), meta_data

    def predict(self, model: Session, images: Image) -> Prediction:
        with threading.Semaphore():
            predictions = model.run(None, {model.get_inputs()[0].name: images})[0]
        return predictions


    def postprocess_predictions(self, predictions: Prediction, meta_data: MetaData, score_threshold=0.25, iou_threshold=0.4) -> List[Face]:
        # (n, 20, 8400) -> (n, 8400, 20)
        predictions = numpy.transpose(predictions, (0, 2, 1))
        predictions = numpy.ascontiguousarray(predictions)
        
        # create batch faces
        batch_faces = []
        for i, pred in enumerate(predictions):
            bbox, score, kps = numpy.split(pred, [4, 5], axis=1)
            ratio, dw, dh = meta_data['ratios'][i], meta_data['dws'][i], meta_data['dhs'][i]
            
            # (x_center, y_center, width, height) -> (x_min, y_min, x_max, y_max)
            # restore to original size
            new_ratio = 1/ratio	
            x_center, y_center, width, height = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            x_min = (x_center - (width / 2) - dw) * new_ratio
            y_min = (y_center - (height / 2) - dh) * new_ratio
            x_max = (x_center + (width / 2) - dw) * new_ratio
            y_max = (y_center + (height / 2) - dh) * new_ratio
            bbox = numpy.stack((x_min, y_min, x_max, y_max), axis=1)
            
            # (x, y, score) -> (x, y)
            # restore to original size
            for i in range(kps.shape[1] // 3):
                kps[:, i * 3] = (kps[:, i * 3] - dw) * new_ratio
                kps[:, i * 3 + 1] = (kps[:, i * 3 + 1] - dh) * new_ratio
            
            # filter
            indices_above_threshold = numpy.where(score > score_threshold)[0]
            bbox = bbox[indices_above_threshold]
            score = score[indices_above_threshold]
            kps = kps[indices_above_threshold]
            
            # nms
            nms_indices = cv2.dnn.NMSBoxes(bbox.tolist(), score.ravel().tolist(), score_threshold, iou_threshold)
            bbox = bbox[nms_indices]
            score = score[nms_indices]
            kps = kps[nms_indices]
            
            # convert to list
            bbox_list = []
            for box in bbox:
                bbox_list.append(numpy.array(
                [
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                ]))
            score_list = score.ravel().tolist()
            kps_list = []
            for keypoints in kps:
                kps_xy = []
                for i in range(0, len(keypoints), 3):
                    kps_xy.append([keypoints[i], keypoints[i+1]])
                kps_list.append(numpy.array(kps_xy))

            batch_faces.append([Face(bbox=bbox, kps=kps, score=score) for bbox, kps, score in zip(bbox_list, kps_list, score_list)])
        return batch_faces

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """
        pre_images = [cv2.imread(image_path)]

        images, meta_data = self.preprocess_images(pre_images)
        predictions = self.predict(self.model, images)
        batch_faces = self.postprocess_predictions(predictions, meta_data)

        shapes = []
        for faces in batch_faces:
            bbox_list = []
            kps_list = []
            for face in faces:
                bbox_list.append(face.bbox)
                kps_list.append(face.kps)
            for bbox, keypoints in zip(bbox_list, kps_list):
                shape = Shape(label=self.classes[0], shape_type="rectangle", flags={})
                shape.add_point(QtCore.QPointF(bbox[0], bbox[1]))
                shape.add_point(QtCore.QPointF(bbox[2], bbox[3]))
                shapes.append(shape)
                
                for i, kp in enumerate(keypoints):
                    shape = Shape(label=str(i), shape_type="point", flags={})
                    shape.add_point(QtCore.QPointF(kp[0], kp[1]))
                    shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
