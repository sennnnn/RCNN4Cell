#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import os
import os.path as osp

from torch import optim

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)

from r4c.config import add_local_config
from r4c.data import MultiClassCellDatasetMapper


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["coco"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError("no Evaluator for the dataset {} with the type {}".format(
            dataset_name, evaluator_type))
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "single_class_nuclei":
            mapper = MultiClassCellDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.TEST_DATASET_MAPPER_NAME == "single_class_nuclei":
            pass
        else:
            mapper = None
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        optimizer_type = cfg.SOLVER.OPTIMIZER

        params = []
        memo = set()
        for n, m in model.named_modules():
            for p_n, p_m in m.named_parameters():
                if not p_m.requires_grad:
                    continue

                if p_m in memo:
                    continue
                # select out params.
                memo.add(p_m)
                if 'backbone' in n:
                    params.append({'params': [p_m], 'lr': cfg.SOLVER.BACKBONE_LR_RATIO * cfg.SOLVER.BASE_LR})
                else:
                    params.append({'params': [p_m]})

        if optimizer_type == "SGD":
            optimizer = optim.SGD(
                params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        elif optimizer_type == "ADAM":
            optimizer = optim.Adam(
                params, lr=cfg.SOLVER.BASE_LR, eps=cfg.SOLVER.EPS, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        elif optimizer_type == 'ADAMW':
            optimizer = optim.AdamW(
                params, lr=cfg.SOLVER.BASE_LR, eps=cfg.SOLVER.EPS, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_local_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if 'OUTPUT_DIR' not in args.opts:
        work_dir_prefix = osp.dirname(args.config_file).replace('configs/', '')
        work_dir_suffix = osp.splitext(osp.basename(args.config_file))[0]
        cfg.OUTPUT_DIR = f'work_dirs/{work_dir_prefix}/{work_dir_suffix}'
        if args.eval_only:
            cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, 'eval')

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
