# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import importlib

__dir__ = os.path.dirname(__file__)

import paddle
from paddle.utils import try_import

sys.path.append(os.path.join(__dir__, ""))

import cv2
import logging
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import pprint
from PIL import Image
import os
import shutil
import random
import paddle.distributed as dist
import tarfile

from tools.infer import predict_system
from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static
from tools.program import preprocess, train

def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


tools = _import_file(
    "tools", os.path.join(__dir__, "tools/__init__.py"), make_importable=True
)
ppocr = importlib.import_module("ppocr", "paddleocr")
ppstructure = importlib.import_module("ppstructure", "paddleocr")
from ppocr.utils.logging import get_logger

from ppocr.utils.utility import (
    check_and_read,
    get_image_file_list,
    alpha_to_color,
    binarize_img,
)
from ppocr.utils.network import (
    maybe_download,
    download_with_progressbar,
    is_link,
    confirm_model_dir_url,
)
from tools.infer.utility import draw_ocr, str2bool, check_gpu
from ppstructure.utility import init_args, draw_structure_result
from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel

logger = get_logger()

__all__ = [
    "PaddleOCR",
    "PPStructure",
    "draw_ocr",
    "draw_structure_result",
    "save_structure_res",
    "download_with_progressbar",
    "to_excel",
]

SUPPORT_DET_MODEL = ["DB"]
SUPPORT_REC_MODEL = ["CRNN", "SVTR_LCNet"]
BASE_DIR = os.path.expanduser("~/.paddleocr/")

DEFAULT_OCR_MODEL_VERSION = "PP-OCRv4"
SUPPORT_OCR_MODEL_VERSION = ["PP-OCR", "PP-OCRv2", "PP-OCRv3", "PP-OCRv4"]
DEFAULT_STRUCTURE_MODEL_VERSION = "PP-StructureV2"
SUPPORT_STRUCTURE_MODEL_VERSION = ["PP-Structure", "PP-StructureV2"]
MODEL_URLS = {
    "OCR": {
        "PP-OCRv4": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar",
                    "train": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_train.tar",
                    "conf":"./libs/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml"
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
                    "train": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar",
                    "conf":"./libs/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml"
                },
                "ml": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar",
                    "train": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar",
                    "conf":"./libs/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
                    "train": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                    "conf":"./libs/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_distill.yml"
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
                    "train": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                    "conf":"./libs/configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml"
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar",
                    "train": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_train.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                    "conf":"./libs/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml"
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar",
                    "train": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_train.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                    "conf":"./libs/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml"
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ta_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/te_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ka_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/arabic_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/devanagari_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCRv3": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
                },
                "ml": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCRv2": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar",
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                }
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCR": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar",
                },
                "structure": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "french": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/french_dict.txt",
                },
                "german": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/german_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
                "structure": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar",
                    "dict_path": "ppocr/utils/dict/table_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
    },
    "STRUCTURE": {
        "PP-Structure": {
            "table": {
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar",
                    "dict_path": "ppocr/utils/dict/table_structure_dict.txt",
                }
            }
        },
        "PP-StructureV2": {
            "table": {
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
                    "dict_path": "ppocr/utils/dict/table_structure_dict.txt",
                },
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar",
                    "dict_path": "ppocr/utils/dict/table_structure_dict_ch.txt",
                },
            },
            "layout": {
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
                    "dict_path": "ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
                },
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar",
                    "dict_path": "ppocr/utils/dict/layout_dict/layout_cdla_dict.txt",
                },
            },
        },
    },
}


def parse_args(mMain=True):
    import argparse

    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default="ch")
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default="ocr")
    parser.add_argument("--savefile", type=str2bool, default=False)
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default="PP-OCRv4",
        help="OCR Model version, the current model support list is as follows: "
        "1. PP-OCRv4/v3 Support Chinese and English detection and recognition model, and direction classifier model"
        "2. PP-OCRv2 Support Chinese detection and recognition model. "
        "3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.",
    )
    parser.add_argument(
        "--structure_version",
        type=str,
        choices=SUPPORT_STRUCTURE_MODEL_VERSION,
        default="PP-StructureV2",
        help="Model version, the current model support list is as follows:"
        " 1. PP-Structure Support en table structure model."
        " 2. PP-StructureV2 Support ch and en table structure model.",
    )

    for action in parser._actions:
        if action.dest in [
            "rec_char_dict_path",
            "table_char_dict_path",
            "layout_dict_path",
        ]:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    latin_lang = [
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
        "french",
        "german",
    ]
    arabic_lang = ["ar", "fa", "ug", "ur"]
    cyrillic_lang = [
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
    ]
    devanagari_lang = [
        "hi",
        "mr",
        "ne",
        "bh",
        "mai",
        "ang",
        "bho",
        "mah",
        "sck",
        "new",
        "gom",
        "sa",
        "bgc",
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert (
        lang in MODEL_URLS["OCR"][DEFAULT_OCR_MODEL_VERSION]["rec"]
    ), "param lang must in {}, but got {}".format(
        MODEL_URLS["OCR"][DEFAULT_OCR_MODEL_VERSION]["rec"].keys(), lang
    )
    if lang == "ch":
        det_lang = "ch"
    elif lang == "structure":
        det_lang = "structure"
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"
    return lang, det_lang


def get_model_config(type, version, model_type, lang):
    if type == "OCR":
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    elif type == "STRUCTURE":
        DEFAULT_MODEL_VERSION = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError

    model_urls = MODEL_URLS[type]
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                "{} models is not support, we only support {}".format(
                    model_type, model_urls[DEFAULT_MODEL_VERSION].keys()
                )
            )
            sys.exit(-1)

    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                "lang {} is not support, we only support {} for {} models".format(
                    lang,
                    model_urls[DEFAULT_MODEL_VERSION][model_type].keys(),
                    model_type,
                )
            )
            sys.exit(-1)
    return model_urls[version][model_type][lang]


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def check_img(img, alpha_color=(255, 255, 255)):
    """
    Check the image data. If it is another type of image file, try to decode it into a numpy array.
    The inference network requires three-channel images, So the following channel conversions are done
        single channel image: Gray to RGB R←Y,G←Y,B←Y
        four channel image: alpha_to_color
    args:
        img: image data
            file format: jpg, png and other image formats that opencv can decode, as well as gif and pdf formats
            storage type: binary image, net image file, local image file
        alpha_color: Background color in images in RGBA format
        return: numpy.array (h, w, 3) or list (p, h, w, 3) (p: page of pdf), boolean, boolean
    """
    flag_gif, flag_pdf = False, False
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, "tmp.jpg")
            img = "tmp.jpg"
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, "rb") as f:
                img_str = f.read()
                img = img_decode(img_str)
            if img is None:
                try:
                    buf = BytesIO()
                    image = BytesIO(img_str)
                    im = Image.open(image)
                    rgb = im.convert("RGB")
                    rgb.save(buf, "jpeg")
                    buf.seek(0)
                    image_bytes = buf.read()
                    data_base64 = str(base64.b64encode(image_bytes), encoding="utf-8")
                    image_decode = base64.b64decode(data_base64)
                    img_array = np.frombuffer(image_decode, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    logger.error("error in loading image:{}".format(image_file))
                    return None, flag_gif, flag_pdf
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None, flag_gif, flag_pdf
    # single channel image array.shape:h,w
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # four channel image array.shape:h,w,c
    if isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 4:
        img = alpha_to_color(img, alpha_color)
    return img, flag_gif, flag_pdf


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert (
            params.ocr_version in SUPPORT_OCR_MODEL_VERSION
        ), "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version
        )
        params.use_gpu = check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        det_model_config = get_model_config("OCR", params.ocr_version, "det", det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, "whl", "det", det_lang),
            det_model_config["url"],
        )
        rec_model_config = get_model_config("OCR", params.ocr_version, "rec", lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, "whl", "rec", lang),
            rec_model_config["url"],
        )
        cls_model_config = get_model_config("OCR", params.ocr_version, "cls", "ch")
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, "whl", "cls"),
            cls_model_config["url"],
        )
        if params.ocr_version in ["PP-OCRv3", "PP-OCRv4"]:
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"
        # download model if using paddle infer
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error("det_algorithm must in {}".format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error("rec_algorithm must in {}".format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config["dict_path"]
            )

        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(
        self,
        img,
        det=True,
        rec=True,
        cls=True,
        bin=False,
        inv=False,
        alpha_color=(255, 255, 255),
        slice={},
    ):
        """
        OCR with PaddleOCR

        args:
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is pure white.
            slice: use sliding window inference for large images, det and rec must be True. Requires int values for slice["horizontal_stride"], slice["vertical_stride"], slice["merge_x_thres"], slice["merge_y_thres] (See doc/doc_en/slice_en.md). Default is {}.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error("When input a list of images, det must be false")
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                "Since the angle classifier is not initialized, it will not be used during the forward process"
            )

        img, flag_gif, flag_pdf = check_img(img, alpha_color)
        # for infer pdf file
        if isinstance(img, list) and flag_pdf:
            if self.page_num > len(img) or self.page_num == 0:
                imgs = img
            else:
                imgs = img[: self.page_num]
        else:
            imgs = [img]

        def preprocess_image(_image):
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                _image = cv2.bitwise_not(_image)
            if bin:
                _image = binarize_img(_image)
            return _image

        if det and rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, rec_res, _ = self.__call__(img, cls, slice)
                if not dt_boxes and not rec_res:
                    ocr_res.append(None)
                    continue
                tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, elapse = self.text_detector(img)
                if dt_boxes.size == 0:
                    ocr_res.append(None)
                    continue
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = preprocess_image(img)
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res


class PPStructure(StructureSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert (
            params.structure_version in SUPPORT_STRUCTURE_MODEL_VERSION
        ), "structure_version must in {}, but get {}".format(
            SUPPORT_STRUCTURE_MODEL_VERSION, params.structure_version
        )
        params.use_gpu = check_gpu(params.use_gpu)
        params.mode = "structure"

        if not params.show_log:
            logger.setLevel(logging.INFO)
        lang, det_lang = parse_lang(params.lang)
        if lang == "ch":
            table_lang = "ch"
        else:
            table_lang = "en"
        if params.structure_version == "PP-Structure":
            params.merge_no_span_structure = False

        # init model dir
        det_model_config = get_model_config("OCR", params.ocr_version, "det", det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, "whl", "det", det_lang),
            det_model_config["url"],
        )
        rec_model_config = get_model_config("OCR", params.ocr_version, "rec", lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, "whl", "rec", lang),
            rec_model_config["url"],
        )
        table_model_config = get_model_config(
            "STRUCTURE", params.structure_version, "table", table_lang
        )
        params.table_model_dir, table_url = confirm_model_dir_url(
            params.table_model_dir,
            os.path.join(BASE_DIR, "whl", "table"),
            table_model_config["url"],
        )
        layout_model_config = get_model_config(
            "STRUCTURE", params.structure_version, "layout", lang
        )
        params.layout_model_dir, layout_url = confirm_model_dir_url(
            params.layout_model_dir,
            os.path.join(BASE_DIR, "whl", "layout"),
            layout_model_config["url"],
        )
        # download model
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.table_model_dir, table_url)
            maybe_download(params.layout_model_dir, layout_url)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config["dict_path"]
            )
        if params.table_char_dict_path is None:
            params.table_char_dict_path = str(
                Path(__file__).parent / table_model_config["dict_path"]
            )
        if params.layout_dict_path is None:
            params.layout_dict_path = str(
                Path(__file__).parent / layout_model_config["dict_path"]
            )
        logger.debug(params)
        super().__init__(params)

    def __call__(
        self,
        img,
        return_ocr_result_in_table=False,
        img_idx=0,
        alpha_color=(255, 255, 255),
    ):
        img, flag_gif, flag_pdf = check_img(img, alpha_color)
        if isinstance(img, list) and flag_pdf:
            res_list = []
            for index, pdf_img in enumerate(img):
                logger.info("processing {}/{} page:".format(index + 1, len(img)))
                res, _ = super().__call__(
                    pdf_img, return_ocr_result_in_table, img_idx=index
                )
                res_list.append(res)
            return res_list
        res, _ = super().__call__(img, return_ocr_result_in_table, img_idx=img_idx)
        return res


def train_paddle(model_type, annoPath, lang):
    from tools.export_model import export_single_model
    # download model if using paddle infer
    params = parse_args(mMain=False)

    if not params.show_log:
        logger.setLevel(logging.INFO)
    lang, det_lang = parse_lang(lang)

    # init model dir
    model_config = get_model_config("OCR", params.ocr_version, model_type, det_lang)
    if model_type == "det":
        model_dir = params.det_model_dir
    elif model_type == "rec":
        model_dir = params.rec_model_dir

    model_dir, det_url = confirm_model_dir_url(
        model_dir,
        os.path.join(BASE_DIR, "whl", model_type, det_lang),
        model_config["train"],
    )
    
    # using custom model
    tar_file_name_list = [".pdparams"]
    if not os.path.exists(
        os.path.join(model_dir, "best_accuracy.pdparams")
    ):
        assert det_url.endswith(".tar"), "Only supports tar compressed package"
        tmp_path = os.path.join(model_dir, det_url.split("/")[-1])
        print("download {} to {}".format(det_url, tmp_path))
        os.makedirs(model_dir, exist_ok=True)
        download_with_progressbar(det_url, tmp_path)
        with tarfile.open(tmp_path, "r") as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if member.name.endswith(tar_file_name):
                        filename = "best_accuracy" + tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(os.path.join(model_dir, filename), "wb") as f:
                    f.write(file.read())
        os.remove(tmp_path)

    genDetRecTrainVal(annoPath)

    config, device, logger, vdl_writer = preprocess(is_train=True, conf= model_config["conf"], data_dir=os.path.join(annoPath, model_type), lang = det_lang)
    # init dist environment
    if config["Global"]["distributed"]:
        dist.init_parallel_env()

    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    config["Global"]["pretrained_model"] = os.path.join(model_dir, "best_accuracy.pdparams")
    if model_type == "rec":
        config["Global"]["character_dict_path"] = os.path.join("./libs", model_config["dict_path"])
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
    
    model.eval()

    save_path = config["Global"]["save_inference_dir"]

    arch_config = config["Architecture"]

    if (
        arch_config["algorithm"] in ["SVTR", "CPPD"]
        and arch_config["Head"]["name"] != "MultiHead"
    ):
        input_shape = config["Eval"]["dataset"]["transforms"][-2]["SVTRRecResizeImg"][
            "image_shape"
        ]
    elif arch_config["algorithm"].lower() == "ABINet".lower():
        rec_rs = [
            c
            for c in config["Eval"]["dataset"]["transforms"]
            if "ABINetRecResizeImg" in c
        ]
        input_shape = rec_rs[0]["ABINetRecResizeImg"]["image_shape"] if rec_rs else None
    else:
        input_shape = None

    if arch_config["algorithm"] in [
        "Distillation",
    ]:  # distillation model
        archs = list(arch_config["Models"].values())
        for idx, name in enumerate(model.model_name_list):
            sub_model_save_path = os.path.join(save_path, name, "inference")
            export_single_model(
                model.model_list[idx], archs[idx], sub_model_save_path, logger
            )
    else:
        save_path = os.path.join(save_path, "inference")
        export_single_model(
            model, arch_config, save_path, logger, input_shape=input_shape
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
trainValTestRatio = "7:2:1"

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
                image_path = image_relative_path
            elif flag == "rec":
                image_path = image_relative_path
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

    splitTrainVal(
                    datasetRootPath,
                    recAbsTrainRootPath,
                    recAbsValRootPath,
                    recAbsTestRootPath,
                    recTrainTxt,
                    recValTxt,
                    recTestTxt,
                    "rec",
                )