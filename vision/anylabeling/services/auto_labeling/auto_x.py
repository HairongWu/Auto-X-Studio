import logging
import os
import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication
from collections import OrderedDict

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from .model import Model
from .types import AutoLabelingResult

import onnxruntime
import threading
import numpy
from collections import namedtuple
from typing import Any, List

from .keypoint_preprocess import *
from .keypoint_postprocess import *

Session = onnxruntime.InferenceSession
Image = numpy.ndarray[Any, Any]
Prediction = numpy.ndarray[Any, Any]
MetaData = dict[str, list[int]]
Bbox = numpy.ndarray[Any, Any]
Kps = numpy.ndarray[Any, Any]
Score = float

score_threshold = 0.25
iou_threshold = 0.4



class Gauge(Model):

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "det_model_path",
            "hfi_model_path",
            "opg_model_path",
            "tpg_model_path",
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
        self.classes = self.config["classes"]

        model_abs_path = self.get_model_abs_path(self.config, "det_model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8 model."
                )
            )
        
        with threading.Lock():
            self.det_model = onnxruntime.InferenceSession(model_abs_path)

        model_abs_path = self.get_model_abs_path(self.config, "hfi_model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8 model."
                )
            )
        
        with threading.Lock():
            self.hfi_model = onnxruntime.InferenceSession(model_abs_path)

        model_abs_path = self.get_model_abs_path(self.config, "opg_model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8 model."
                )
            )
        
        with threading.Lock():
            self.opg_model = onnxruntime.InferenceSession(model_abs_path)

        model_abs_path = self.get_model_abs_path(self.config, "tpg_model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8 model."
                )
            )
        
        with threading.Lock():
            self.tpg_model = onnxruntime.InferenceSession(model_abs_path)

    def preprocess_det(self, images: List[Image], input_size=(640, 640)) -> tuple[numpy.ndarray[Any, Any], MetaData]:
        preprocessed = []
        meta_data = {'ratios': [], 'dws': [], 'dhs': []}
        for image in images:
            # resize
            shape = image.shape[:2]
            ratio = (input_size[0] / shape[0], input_size[1] / shape[1])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, input_size)
            # normalize
            image = image.astype(numpy.float32) / 255.0
            
            # RGB -> BGR
            # image = image[..., ::-1]
            
            # HWC -> CHW
            image = image.transpose((2, 0, 1))
            
            preprocessed.append(image)
            meta_data['ratios'].append(ratio)

        return numpy.ascontiguousarray(numpy.array(preprocessed)), meta_data

    def predict(self, model: Session, images: Image) -> Prediction:
        with threading.Semaphore():
            predictions = model.run(None, {model.get_inputs()[0].name: images})[0]
        return predictions


    def postprocess_det(self, predictions: Prediction, meta_data: MetaData, score_threshold=0.25, iou_threshold=0.4) -> List:
        # (n, 20, 8400) -> (n, 8400, 20)
        #predictions = numpy.transpose(predictions, (0, 2, 1))
        predictions = numpy.ascontiguousarray(predictions)
        # create batch faces
        batch_faces = []
        classes = []
        for i, pred in enumerate(predictions):
            
            bbox, score, cls = numpy.hsplit(pred, [4, 5])
            
            ratio = meta_data['ratios'][i]
            
            # (x_center, y_center, width, height) -> (x_min, y_min, x_max, y_max)
            # restore to original size
            new_ratio = (1/ratio[1], 1/ratio[0])
            x_center, y_center, width, height = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            x_min = (x_center) * new_ratio[0]
            y_min = (y_center) * new_ratio[1]
            x_max = width * new_ratio[0]
            y_max = height* new_ratio[1]
            bbox = numpy.stack((x_min, y_min, x_max, y_max), axis=1)
            
            # (x, y, score) -> (x, y)
            # restore to original size
            # for i in range(kps.shape[1] // 3):
            #     kps[:, i * 3] = (kps[:, i * 3] - dw) * new_ratio
            #     kps[:, i * 3 + 1] = (kps[:, i * 3 + 1] - dh) * new_ratio
            
            # filter
            indices_above_threshold = numpy.where(score > score_threshold)[0]
            bbox = bbox[indices_above_threshold]
            score = score[indices_above_threshold]
            cls = cls[indices_above_threshold]
            
            # nms
            nms_indices = cv2.dnn.NMSBoxes(bbox.tolist(), score.ravel().tolist(), score_threshold, iou_threshold)
            bbox = bbox[nms_indices]
            score = score[nms_indices]
            cls = cls[nms_indices]

            classes.append(int(cls))
            batch_faces.append(bbox)
        return batch_faces, classes

    def process_rec(self, image, rec_model):
        # load preprocess transforms
        pre_operators_key = [
            TopDownEvalAffine(trainsize=[288, 384]),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            Permute()
        ]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_info = {
            "im_shape": np.array(
                img.shape[:2], dtype=np.float32),
            "scale_factor": np.array(
                [1., 1.], dtype=np.float32)
        }

        for i, op in enumerate(pre_operators_key):
            img, img_info = op(img, img_info)

        outname = [output.name for output in rec_model.get_outputs()]
        np_heatmap = rec_model.run(outname, {rec_model.get_inputs()[0].name: [img]})[0]
        # np_heatmap = OrderedDict(zip(outname, np_heatmap))

        imshape = np.array(
                image.shape[:2], dtype=np.float32)
        center = np.round(imshape / 2.)
        scale = imshape / 200.

        keypoint_postprocess = HRNetPostProcess(use_dark=True)
        kpts, scores = keypoint_postprocess(np_heatmap, [center], [scale])

        return kpts
        
    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """
        pre_images = [cv2.imread(image_path)]
        
        images, meta_data = self.preprocess_det(pre_images)
        predictions = self.predict(self.det_model, images)
        batch_faces, classes = self.postprocess_det(predictions, meta_data)
        
        shapes = []

        for img, faces, cls in zip(pre_images, batch_faces, classes):

            for bbox in faces:
                
                shape = Shape(label=self.classes[cls], shape_type="rectangle", flags={})
                shape.add_point(QtCore.QPointF(bbox[0], bbox[1]))
                shape.add_point(QtCore.QPointF(bbox[2], bbox[3]))
                shapes.append(shape)

                rec_model = None
                if self.classes[cls] == 'one pointer gauge':
                    rec_model = self.opg_model
                if self.classes[cls] == 'two pointer gauge':
                    rec_model = self.tpg_model
                if self.classes[cls] == 'high flow indicator':
                    rec_model = self.hfi_model

                keypoints = self.process_rec(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]),:], rec_model)

                # keypoints = face.kps
                for i, kp in enumerate(keypoints[0]):
                    shape = Shape(label=str(i), shape_type="point", flags={})
                    shape.add_point(QtCore.QPointF(kp[0]+bbox[0], kp[1]+bbox[1]))
                    shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.det_model
        del self.opg_model
        del self.tpg_model
        del self.hfi_model
