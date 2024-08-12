try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import time
import datetime
import os

from libs.utils import newIcon
from .paddleocr import genDetRecTrainVal, train_paddle, preprocess, set_seed

BB = QDialogButtonBox


class Worker(QThread):
    progressBarValue = pyqtSignal(int)
    listValue = pyqtSignal(str)
    endsignal = pyqtSignal(int, str)
    handle = 0

    def __init__(self, annoPath, lang, mainThread, model):
        super(Worker, self).__init__()
        self.annoPath = annoPath
        self.lang = lang
        self.mainThread = mainThread
        self.model = model
        self.setStackSize(1024 * 1024)

    def run(self):
        try:
            findex = 0
            self.listValue.emit("genDetRecTrainVal")
            genDetRecTrainVal(self.annoPath)
            findex += 1
            self.progressBarValue.emit(findex)

            self.listValue.emit("preprocess")
            config, device, logger, vdl_writer = preprocess(is_train=True, conf= "./libs/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml", data_dir=self.annoPath+"/det", lang=self.lang)
            findex += 1
            self.progressBarValue.emit(findex)
            seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
            set_seed(seed)
            self.listValue.emit("training...")
            train_paddle(config, device, logger, vdl_writer, seed)

            findex += 1
            self.progressBarValue.emit(findex)

            lg_idx = {
                    "ch":"./libs/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml",
                    "en":"./libs/configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml",
                    "french":"./libs/configs/rec/multi_language/rec_french_lite_train.yml",
                    "german":"./libs/configs/rec/multi_language/rec_german_lite_train.yml",
                    "korean":"./libs/configs/rec/multi_language/rec_korean_lite_train.yml",
                    "japan":"./libs/configs/rec/multi_language/rec_japan_lite_train.yml",
            }

            self.listValue.emit("preprocess")
            config, device, logger, vdl_writer = preprocess(is_train=True, conf= lg_idx[self.lang], data_dir=self.annoPath + "/rec", lang=self.lang)
            findex += 1
            self.progressBarValue.emit(findex)
            seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
            set_seed(seed)

            self.listValue.emit("training...")
            train_paddle(config, device, logger, vdl_writer, seed)
            findex += 1
            self.progressBarValue.emit(findex)
            self.listValue.emit("finished")
            self.endsignal.emit(0, "readAll")
            self.exec()
        except Exception as e:
            print(e)
            raise


class TrainDialog(QDialog):
    def __init__(
        self, text="Enter object label", parent=None, annos=None, lang = "en"
    ):
        super(TrainDialog, self).__init__(parent)
        self.setFixedWidth(1000)
        self.parent = parent
        self.annos = annos
        self.lang = lang
        self.lender = 5
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

        self.thread_1 = Worker(self.annos, self.lang, self.parent, "paddle")
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
