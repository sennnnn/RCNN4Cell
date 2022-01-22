import os.path as osp
import copy

import cv2
import numpy as np
import torch
from PIL import Image

import detectron2.data.detection_utils as utils
from detectron2.config import configurable
from detectron2.structures import Instances, BitMasks

__all__ = ["MultiClassCellDatasetMapper"]


def read_image(path):
    _, suffix = osp.splitext(osp.basename(path))
    if suffix == '.tif':
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif suffix == '.npy':
        img = np.load(path)
    else:
        img = Image.open(path)
        img = np.array(img)

    return img


def to_one_hot(mask, num_classes):
    ret = np.zeros((num_classes, *mask.shape))
    for i in range(num_classes):
        ret[i, mask == i] = 1

    return ret


def assign_sem_class_to_insts(inst_seg, sem_seg, num_classes):
    inst_id_list = list(np.unique(inst_seg))

    if 0 not in inst_id_list:
        inst_id_list.insert(0, 0)

    sem_seg_one_hot = to_one_hot(sem_seg, num_classes)

    # Remove background class
    inst_id_list_per_class = {}
    for inst_id in inst_id_list:
        inst_mask = (inst_seg == inst_id).astype(np.uint8)

        tp = np.sum(inst_mask * sem_seg_one_hot, axis=(-2, -1))

        if np.sum(tp[1:]) > 0 and inst_id != 0:
            belong_sem_id = np.argmax(tp[1:]) + 1
        else:
            belong_sem_id = 0

        if belong_sem_id not in inst_id_list_per_class:
            inst_id_list_per_class[belong_sem_id] = [inst_id]
        else:
            inst_id_list_per_class[belong_sem_id].append(inst_id)

    return inst_id_list_per_class


class MultiClassCellDatasetMapper:

    @configurable
    def __init__(
        self,
        is_train,
        *,
        num_classes,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
        """
        # fmt: off
        self.is_train = is_train
        self.num_classes = num_classes
        # fmt: on

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):

        ret = {"is_train": is_train, "num_classes": cfg.INPUT.NUM_CLASSES}

        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
            for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.instance_mask_format)

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format='RGB')
        sem_seg = read_image(dataset_dict['sem_ann_file_name'])
        inst_seg = read_image(dataset_dict['inst_ann_file_name'])

        # aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        # transforms = self.augmentations(aug_input)
        # image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        dataset_dict['height'] = image_shape[0]
        dataset_dict['width'] = image_shape[1]
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # if sem_seg_gt is not None:
        #     dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        inst_ids_per_class = assign_sem_class_to_insts(inst_seg, sem_seg, self.num_classes)

        assert len(
            inst_ids_per_class[0]
        ) == 1, 'It may be wrong when a cell region belong to background class except it is background region.'

        # construct instances for mask r-cnn instance segmentation
        instances = Instances(image_shape)
        masks = []
        classes = []
        for cat_id in inst_ids_per_class.keys():
            if cat_id == 0:
                continue
            inst_ids = inst_ids_per_class[cat_id]
            for inst_id in inst_ids:
                masks.append(inst_seg == inst_id)
                classes.append(cat_id - 1)
        if len(masks) == 0:
            instances.gt_masks = BitMasks(torch.zeros((0, *image_shape)))
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        else:
            masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks]))
            classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_masks = masks
            instances.gt_classes = classes
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
