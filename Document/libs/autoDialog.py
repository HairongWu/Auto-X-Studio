try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import time
import datetime
import json
import cv2
import numpy as np
import copy

from libs.utils import newIcon

from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import (
    draw_ocr_box_txt,
    get_rotate_crop_image,
    get_minarea_rect_crop,
    slice_generator,
    merge_fragmented,
)
logger = get_logger()

BB = QDialogButtonBox

class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self

class TextSystem(object):
    def __init__(self, args):

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def __call__(self, img, cls=True, slice={}):
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        if slice:
            slice_gen = slice_generator(
                img,
                horizontal_stride=slice["horizontal_stride"],
                vertical_stride=slice["vertical_stride"],
            )
            elapsed = []
            dt_slice_boxes = []
            for slice_crop, v_start, h_start in slice_gen:
                dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                if dt_boxes.size:
                    dt_boxes[:, :, 0] += h_start
                    dt_boxes[:, :, 1] += v_start
                    dt_slice_boxes.append(dt_boxes)
                    elapsed.append(elapse)
            dt_boxes = np.concatenate(dt_slice_boxes)

            dt_boxes = merge_fragmented(
                boxes=dt_boxes,
                x_threshold=slice["merge_x_thres"],
                y_threshold=slice["merge_y_thres"],
            )
            elapse = sum(elapsed)
        else:
            dt_boxes, elapse = self.text_detector(img)

        time_dict["det"] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict
        else:
            logger.debug(
                "dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse)
            )
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
            time_dict["cls"] = elapse
            logger.debug(
                "cls num  : {}, elapsed : {}".format(len(img_crop_list), elapse)
            )
        if len(img_crop_list) > 1000:
            logger.debug(
                f"rec crops num: {len(img_crop_list)}, time and memory cost may be large."
            )

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
        
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict["all"] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

class Worker(QThread):
    progressBarValue = pyqtSignal(int)
    listValue = pyqtSignal(str)
    endsignal = pyqtSignal(int, str)
    handle = 0

    def __init__(self, lang, mImgList, mainThread, model):
        super(Worker, self).__init__()
        self.lang = lang
        self.mImgList = mImgList
        self.mainThread = mainThread
        self.model = model
        self.setStackSize(1024 * 1024)

    def run(self):
        try:
            args = AttrDict()
        
            args['det_model_dir'] = "E:\\app\\model_pool\\ch_PP-OCRv4_det_server_infer\\"
            args['cls_model_dir'] = "E:\\app\\model_pool\\ch_ppocr_mobile_v2.0_cls_infer\\"

            lg_idx = {
                    "ch":"E:\\app\\model_pool\\ch_PP-OCRv4_rec_server_infer\\",
                    "en":"E:\\app\\model_pool\\ch_PP-OCRv4_rec_server_infer\\",
                    "french":"E:\\app\\model_pool\\",
                    "german":"E:\\app\\model_pool\\",
                    "korean":"E:\\app\\model_pool\\korean_PP-OCRv3_rec_infer\\",
                    "japan":"E:\\app\\model_pool\\japan_PP-OCRv3_rec_infer\\",
            }
            args['rec_model_dir'] = lg_idx[self.lang]

            lg_idx = {
                    "ch":"ppocr/utils/chinese_cht_dict.txt",
                    "en":"ppocr/utils/dict/japan_dict.txt",
                    "french":"ppocr/utils/dict/french_dict.txt",
                    "german":"ppocr/utils/dict/german_dict.txt",
                    "korean":"ppocr/utils/dict/korean_dict.txt",
                    "japan":"ppocr/utils/dict/japan_dict.txt",
            }

            args['rec_char_dict_path'] = lg_idx[self.lang]

            args['use_tensorrt'] = False
            args['use_onnx'] = False
            args['drop_score'] = 0.5
            args['use_gpu'] = True
            args['gpu_mem'] = 500
            args['gpu_id'] = 0
            args['return_word_box'] = False
            args['benchmark'] = False

            args['det_box_type'] = 'quad'
            args['det_algorithm'] = "DB"
            args['det_limit_side_len'] = 960
            args['det_limit_type'] = 'max'

            args['det_db_thresh'] = 0.3
            args['det_db_box_thresh'] = 0.6
            args['det_db_unclip_ratio'] = 1.5
            args['use_dilation'] = False
            args['det_db_score_mode'] = 'fast'

            args['det_east_score_thresh'] = 0.8
            args['det_east_cover_thresh'] = 0.1
            args['det_east_nms_thresh'] = 0.2

            args['det_sast_score_thresh'] = 0.5
            args['det_sast_nms_thresh'] = 0.2

            args['det_pse_thresh'] = 0
            args['det_pse_box_thresh'] = 0.85
            args['det_pse_min_area'] = 16
            args['det_pse_scale'] = 1

            args['scales'] = [8, 16, 32]
            args['alpha'] = 1.0
            args['beta'] = 1.0
            args['fourier_degree'] = 5

            args['rec_algorithm'] = "SVTR_LCNet"
            args['rec_image_shape'] = "3, 48, 320"
            args['rec_batch_num'] = 6
            args['use_space_char'] = True
            args['max_text_length'] = 25
            args['rec_image_inverse'] = True

            args['use_angle_cls'] = True
            args['cls_image_shape'] = "3, 48, 192"
            args['cls_batch_num'] = 6
            args['cls_thresh'] = 0.9
            args['label_list'] = ["0", "180"]

            text_sys = TextSystem(args)

            findex = 0
            for Imgpath in self.mImgList:
                if self.handle == 0:
                    self.listValue.emit(Imgpath)
                    if self.model == "paddle":
                        h, w, _ = cv2.imdecode(
                            np.fromfile(Imgpath, dtype=np.uint8), 1
                        ).shape
                        if h > 32 and w > 32:
                            img = cv2.imread(Imgpath)
                            dt_boxes, rec_res, time_dict = text_sys(img)
                            self.result_dic = [
                                {
                                    "transcription": rec_res[i][0],
                                    "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                                }
                                for i in range(len(dt_boxes))
                            ]

                        else:
                            print(
                                "The size of", Imgpath, "is too small to be recognised"
                            )
                            self.result_dic = None

                    # 结果保存
                    if self.result_dic is None or len(self.result_dic) == 0:
                        print("Can not recognise file", Imgpath)
                        pass
                    else:
                        strs = ""
                        for res in self.result_dic:
                            strs += (
                                "Transcription: "
                                + res['transcription']
                                + " Probability: "
                                + str(0)
                                + " Location: "
                                + json.dumps(res['points'])
                                + "\n"
                            )
                        # Sending large amounts of data repeatedly through pyqtSignal may affect the program efficiency
                        self.listValue.emit(strs)
                        self.mainThread.result_dic = self.result_dic
                        self.mainThread.filePath = Imgpath
                        # 保存
                        self.mainThread.saveFile(mode="Auto")
                    findex += 1
                    self.progressBarValue.emit(findex)
                else:
                    break
            self.endsignal.emit(0, "readAll")
            self.exec()
        except Exception as e:
            print(e)
            raise


class AutoDialog(QDialog):
    def __init__(
        self, text="Enter object label", parent=None, lang= None, mImgList=None, lenbar=0
    ):
        super(AutoDialog, self).__init__(parent)
        self.setFixedWidth(1000)
        self.parent = parent

        self.mImgList = mImgList
        self.lender = lenbar
        self.pb = QProgressBar()
        self.pb.setRange(0, self.lender)
        self.pb.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.pb)
        self.model = "paddle"
        self.listWidget = QListWidget(self)
        layout.addWidget(self.listWidget)

        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon("done"))
        bb.button(BB.Cancel).setIcon(newIcon("undo"))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        bb.button(BB.Ok).setEnabled(False)

        self.setLayout(layout)
        # self.setWindowTitle("自动标注中")
        self.setWindowModality(Qt.ApplicationModal)

        # self.setWindowFlags(Qt.WindowCloseButtonHint)

        self.thread_1 = Worker(lang, self.mImgList, self.parent, "paddle")
        self.thread_1.progressBarValue.connect(self.handleProgressBarSingal)
        self.thread_1.listValue.connect(self.handleListWidgetSingal)
        self.thread_1.endsignal.connect(self.handleEndsignalSignal)
        self.time_start = time.time()  # save start time

    def handleProgressBarSingal(self, i):
        self.pb.setValue(i)

        # calculate time left of auto labeling
        avg_time = (
            time.time() - self.time_start
        ) / i  # Use average time to prevent time fluctuations
        time_left = str(datetime.timedelta(seconds=avg_time * (self.lender - i))).split(
            "."
        )[
            0
        ]  # Remove microseconds
        self.setWindowTitle("PPOCRLabel  --  " + f"Time Left: {time_left}")  # show

    def handleListWidgetSingal(self, i):
        self.listWidget.addItem(i)
        titem = self.listWidget.item(self.listWidget.count() - 1)
        self.listWidget.scrollToItem(titem)

    def handleEndsignalSignal(self, i, str):
        if i == 0 and str == "readAll":
            self.buttonBox.button(BB.Ok).setEnabled(True)
            self.buttonBox.button(BB.Cancel).setEnabled(False)

    def reject(self):
        print("reject")
        self.thread_1.handle = -1
        self.thread_1.quit()
        # del self.thread_1
        # if self.thread_1.isRunning():
        #     self.thread_1.terminate()
        # self.thread_1.quit()
        # super(AutoDialog,self).reject()
        while not self.thread_1.isFinished():
            pass
        self.accept()

    def validate(self):
        self.accept()

    def postProcess(self):
        try:
            self.edit.setText(self.edit.text().trimmed())
            # print(self.edit.text())
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            self.edit.setText(self.edit.text())
            print(self.edit.text())

    def popUp(self):
        self.thread_1.start()
        return 1 if self.exec_() else None

    def closeEvent(self, event):
        print("???")
        # if self.thread_1.isRunning():
        #     self.thread_1.quit()
        #
        #     # self._thread.terminate()
        # # del self.thread_1
        # super(AutoDialog, self).closeEvent(event)
        self.reject()
