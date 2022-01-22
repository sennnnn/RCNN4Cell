# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor, default_setup
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures import Instances

from r4c.config import add_local_config

# constants
WINDOW_NAME = "COCO detections"


def unravel_instances(instances):
    classes = instances.pred_classes.cpu().numpy()
    masks = instances.pred_masks.cpu().numpy()
    h, w = instances.image_size
    canvas = np.zeros((h, w), dtype=np.uint8)

    for class_id, mask in zip(classes, masks):
        canvas[mask > 0] = class_id

    return canvas


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_local_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/conic/mask_rcnn-c4.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)

    path = 'datasets/conic/train/0.png'

    img = read_image(path, format="BGR")
    res = predictor(img)

    sem_res = unravel_instances(res['instances'])

    import matplotlib.pyplot as plt
    plt.imshow(sem_res)
    plt.savefig('2.png')

    print(res)
