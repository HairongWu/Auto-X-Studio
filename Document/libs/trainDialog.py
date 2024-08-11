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
import shutil
import random

from libs.utils import newIcon

import paddle
import paddle.distributed as dist

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static
from tools.program import *

dist.get_world_size()

BB = QDialogButtonBox

def main(config, device, logger, vdl_writer, seed):
    # init dist environment
    if config["Global"]["distributed"]:
        dist.init_parallel_env()

    global_config = config["Global"]

    # build dataloader
    set_signal_handlers()
    train_dataloader = build_dataloader(config, "Train", device, logger, seed)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n"
            + "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            + "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config["Eval"]:
        valid_dataloader = build_dataloader(config, "Eval", device, logger, seed)
    else:
        valid_dataloader = None
    step_pre_epoch = len(train_dataloader)

    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num
                    # update SARLoss params
                    if (
                        list(config["Loss"]["loss_config_list"][-1].keys())[0]
                        == "DistillationSARLoss"
                    ):
                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"][
                            "ignore_index"
                        ] = (char_num + 1)
                        out_channels_list["SARLabelDecode"] = char_num + 2
                    elif any(
                        "DistillationNRTRLoss" in d
                        for d in config["Loss"]["loss_config_list"]
                    ):
                        out_channels_list["NRTRLabelDecode"] = char_num + 3

                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
            # update SARLoss params
            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                        "ignore_index": char_num + 1
                    }
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                        char_num + 1
                    )
                out_channels_list["SARLabelDecode"] = char_num + 2
            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":
                out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

        if config["PostProcess"]["name"] == "SARLabelDecode":  # for SAR model
            config["Loss"]["ignore_index"] = char_num - 1

    model = build_model(config["Architecture"])

    use_sync_bn = config["Global"].get("use_sync_bn", False)
    if use_sync_bn:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("convert_sync_batchnorm")

    model = apply_to_static(model, config, logger)

    # build loss
    loss_class = build_loss(config["Loss"])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    # build metric
    eval_class = build_metric(config["Metric"])

    logger.info("train dataloader has {} iters".format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info("valid dataloader has {} iters".format(len(valid_dataloader)))

    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")
    amp_dtype = config["Global"].get("amp_dtype", "float16")
    amp_custom_black_list = config["Global"].get("amp_custom_black_list", [])
    amp_custom_white_list = config["Global"].get("amp_custom_white_list", [])
    if use_amp:
        AMP_RELATED_FLAGS_SETTING = {
            "FLAGS_max_inplace_grad_add": 8,
        }
        if paddle.is_compiled_with_cuda():
            AMP_RELATED_FLAGS_SETTING.update(
                {
                    "FLAGS_cudnn_batchnorm_spatial_persistent": 1,
                    "FLAGS_gemm_use_half_precision_compute_type": 0,
                }
            )
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get(
            "use_dynamic_loss_scaling", False
        )
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )
        if amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=amp_level,
                master_weight=True,
                dtype=amp_dtype,
            )
    else:
        scaler = None

    # load pretrain model
    pre_best_model_dict = load_model(
        config, model, optimizer, config["Architecture"]["model_type"]
    )

    if config["Global"]["distributed"]:
        model = paddle.DataParallel(model)
    # start train
    train(
        config,
        train_dataloader,
        valid_dataloader,
        device,
        model,
        loss_class,
        optimizer,
        lr_scheduler,
        post_process_class,
        eval_class,
        pre_best_model_dict,
        logger,
        step_pre_epoch,
        vdl_writer,
        scaler,
        amp_level,
        amp_custom_black_list,
        amp_custom_white_list,
        amp_dtype,
    )

def isCreateOrDeleteFolder(path, flag):
    flagPath = os.path.join(path, flag)

    if os.path.exists(flagPath):
        shutil.rmtree(flagPath)

    os.makedirs(flagPath)
    flagAbsPath = os.path.abspath(flagPath)
    return flagAbsPath

detLabelFileName = "Label.txt"
recLabelFileName = "rec_gt.txt"
recImageDirName = "crop_img"
trainValTestRatio = "6:2:2"

def splitTrainVal(
    root,
    abs_train_root_path,
    abs_val_root_path,
    abs_test_root_path,
    train_txt,
    val_txt,
    test_txt,
    flag,
):
    data_abs_path = os.path.abspath(root)
    label_file_name = detLabelFileName if flag == "det" else recLabelFileName
    label_file_path = os.path.join(data_abs_path, label_file_name)

    with open(label_file_path, "r", encoding="UTF-8") as label_file:
        label_file_content = label_file.readlines()
        random.shuffle(label_file_content)
        label_record_len = len(label_file_content)

        for index, label_record_info in enumerate(label_file_content):
            image_relative_path, image_label = label_record_info.split("\t")
            image_name = os.path.basename(image_relative_path)

            if flag == "det":
                image_path = os.path.join(data_abs_path, image_name)
            elif flag == "rec":
                image_path = os.path.join(
                    data_abs_path, recImageDirName, image_name
                )

            train_val_test_ratio = trainValTestRatio.split(":")
            train_ratio = eval(train_val_test_ratio[0]) / 10
            val_ratio = train_ratio + eval(train_val_test_ratio[1]) / 10
            cur_ratio = index / label_record_len

            if cur_ratio < train_ratio:
                image_copy_path = os.path.join(abs_train_root_path, image_name)
                shutil.copy(image_path, image_copy_path)
                train_txt.write("{}\t{}".format(image_copy_path, image_label))
            elif cur_ratio >= train_ratio and cur_ratio < val_ratio:
                image_copy_path = os.path.join(abs_val_root_path, image_name)
                shutil.copy(image_path, image_copy_path)
                val_txt.write("{}\t{}".format(image_copy_path, image_label))
            else:
                image_copy_path = os.path.join(abs_test_root_path, image_name)
                shutil.copy(image_path, image_copy_path)
                test_txt.write("{}\t{}".format(image_copy_path, image_label))


def removeFile(path):
    if os.path.exists(path):
        os.remove(path)

def genDetRecTrainVal(datasetRootPath = "../train_data/"):
    detRootPath = os.path.join(datasetRootPath, "det")
    recRootPath = os.path.join(datasetRootPath, "rec")

    detAbsTrainRootPath = isCreateOrDeleteFolder(detRootPath, "train")
    detAbsValRootPath = isCreateOrDeleteFolder(detRootPath, "val")
    detAbsTestRootPath = isCreateOrDeleteFolder(detRootPath, "test")
    recAbsTrainRootPath = isCreateOrDeleteFolder(recRootPath, "train")
    recAbsValRootPath = isCreateOrDeleteFolder(recRootPath, "val")
    recAbsTestRootPath = isCreateOrDeleteFolder(recRootPath, "test")

    removeFile(os.path.join(detRootPath, "train.txt"))
    removeFile(os.path.join(detRootPath, "val.txt"))
    removeFile(os.path.join(detRootPath, "test.txt"))
    removeFile(os.path.join(recRootPath, "train.txt"))
    removeFile(os.path.join(recRootPath, "val.txt"))
    removeFile(os.path.join(recRootPath, "test.txt"))

    detTrainTxt = open(
        os.path.join(detRootPath, "train.txt"), "a", encoding="UTF-8"
    )
    detValTxt = open(os.path.join(detRootPath, "val.txt"), "a", encoding="UTF-8")
    detTestTxt = open(os.path.join(detRootPath, "test.txt"), "a", encoding="UTF-8")
    recTrainTxt = open(
        os.path.join(recRootPath, "train.txt"), "a", encoding="UTF-8"
    )
    recValTxt = open(os.path.join(recRootPath, "val.txt"), "a", encoding="UTF-8")
    recTestTxt = open(os.path.join(recRootPath, "test.txt"), "a", encoding="UTF-8")

    splitTrainVal(
        datasetRootPath,
        detAbsTrainRootPath,
        detAbsValRootPath,
        detAbsTestRootPath,
        detTrainTxt,
        detValTxt,
        detTestTxt,
        "det",
    )

    for root, dirs, files in os.walk(datasetRootPath):
        for dir in dirs:
            if dir == "crop_img":
                splitTrainVal(
                    root,
                    recAbsTrainRootPath,
                    recAbsValRootPath,
                    recAbsTestRootPath,
                    recTrainTxt,
                    recValTxt,
                    recTestTxt,
                    "rec",
                )
            else:
                continue
        break

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
            config, device, logger, vdl_writer = preprocess(is_train=True, conf= "./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml", data_dir=self.annoPath+"/det", lang=self.lang)
            findex += 1
            self.progressBarValue.emit(findex)
            seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
            set_seed(seed)
            self.listValue.emit("training...")
            main(config, device, logger, vdl_writer, seed)

            findex += 1
            self.progressBarValue.emit(findex)

            lg_idx = {
                    "ch":"./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml",
                    "en":"./configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml",
                    "french":"./configs/rec/multi_language/rec_french_lite_train.yml",
                    "german":"./configs/rec/multi_language/rec_german_lite_train.yml",
                    "korean":"./configs/rec/multi_language/rec_korean_lite_train.yml",
                    "japan":"./configs/rec/multi_language/rec_japan_lite_train.yml",
            }
            self.listValue.emit("preprocess")
            config, device, logger, vdl_writer = preprocess(is_train=True, conf= lg_idx[self.lang], data_dir=self.annoPath + "/rec", lang=self.lang)
            findex += 1
            self.progressBarValue.emit(findex)
            seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
            set_seed(seed)
            self.listValue.emit("training...")
            main(config, device, logger, vdl_writer, seed)
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
