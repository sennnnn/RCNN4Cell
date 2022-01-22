# Copyright (c) Facebook, Inc. and its affiliates.
import os
import os.path as osp

from detectron2.data import DatasetCatalog, MetadataCatalog

from pycocotools import coco

MONUSEG_CATEGORIES = [
    {
        "color": [255, 2, 255],
        "name": "nuclei",
        "id": 1,
        "trainId": 0
    },
    {
        'color': [255, 255, 255],
        'name': 'edge',
        'id': 2,
        'trainId': 1
    },
]

CLASSES = ('nuclei', 'edge')
LABEL_MAP = {1: 0, 2: 1}

# dataset split
only_train_split_dict = {
    'train': [
        'TCGA-A7-A13E-01Z-00-DX1', 'TCGA-A7-A13F-01Z-00-DX1', 'TCGA-AR-A1AK-01Z-00-DX1', 'TCGA-B0-5711-01Z-00-DX1',
        'TCGA-HE-7128-01Z-00-DX1', 'TCGA-HE-7129-01Z-00-DX1', 'TCGA-18-5592-01Z-00-DX1', 'TCGA-38-6178-01Z-00-DX1',
        'TCGA-49-4488-01Z-00-DX1', 'TCGA-G9-6336-01Z-00-DX1', 'TCGA-G9-6348-01Z-00-DX1', 'TCGA-G9-6356-01Z-00-DX1'
    ],
    'val':
    ['TCGA-AR-A1AS-01Z-00-DX1', 'TCGA-HE-7130-01Z-00-DX1', 'TCGA-50-5931-01Z-00-DX1', 'TCGA-G9-6363-01Z-00-DX1'],
    'test1': [
        'TCGA-E2-A1B5-01Z-00-DX1', 'TCGA-E2-A14V-01Z-00-DX1', 'TCGA-B0-5710-01Z-00-DX1', 'TCGA-B0-5698-01Z-00-DX1',
        'TCGA-21-5784-01Z-00-DX1', 'TCGA-21-5786-01Z-00-DX1', 'TCGA-CH-5767-01Z-00-DX1', 'TCGA-G9-6362-01Z-00-DX1'
    ],
    'test2': [
        'TCGA-DK-A2I6-01A-01-TS1', 'TCGA-G2-A2EK-01A-02-TSB', 'TCGA-AY-A8YK-01A-01-TS1', 'TCGA-NH-A8F7-01A-01-TS1',
        'TCGA-KB-A93J-01A-01-TS1', 'TCGA-RD-A8N9-01A-01-TS1'
    ]
}


def load_monuseg(gt_json_path, img_dir):
    gt_coco_obj = coco.COCO(gt_json_path)
    img_ids = gt_coco_obj.getImgIds()
    imgs = gt_coco_obj.loadImgs(img_ids)

    dataset_dicts = []
    for img_id, img in zip(img_ids, imgs):
        dataset_dict = {}
        dataset_dict['file_name'] = osp.join(img_dir, img['file_name'])
        dataset_dict['height'] = img['height']
        dataset_dict['width'] = img['width']
        dataset_dict['image_id'] = img['id']
        ann_ids = gt_coco_obj.getAnnIds(imgIds=[img_id])
        dataset_dict['annotations'] = gt_coco_obj.loadAnns(ann_ids)

        dataset_dicts.append(dataset_dict)

    return dataset_dicts


def register_monuseg(root):
    root = osp.join(root, "monuseg_coco")

    for name, json_path in [("train", "instances_train.json"), ("val", "instances_val.json")]:
        image_dir = osp.join(root, "images", name)
        gt_json_path = osp.join(root, "annotations", json_path)
        gt_map_path = osp.join(root, 'nucleimaps')

        if name == 'train':
            img_list = only_train_split_dict['train'] + only_train_split_dict['val']
        elif name == 'val':
            img_list = only_train_split_dict['test1'] + only_train_split_dict['test2']

        name = f"monuseg_onlytrain_{name}"

        DatasetCatalog.register(name, lambda x=gt_json_path, y=image_dir: load_monuseg(x, y))

        MetadataCatalog.get(name).set(
            image_root=image_dir,
            instances_path=gt_json_path,
            maps_path=gt_map_path,
            img_list=img_list,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_monuseg(_root)
