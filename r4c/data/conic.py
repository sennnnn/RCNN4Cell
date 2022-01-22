import os
import os.path as osp

from detectron2.data import DatasetCatalog, MetadataCatalog

# yapf:disable
CONIC_CATEGORIES = [
    {'color': [255, 0, 0], 'name': 'neutrophil', 'id': 1, 'trainId': 0},
    {'color': [0, 255, 0], 'name': 'epithelial', 'id': 2, 'trainId': 1},
    {'color': [0, 0, 255], 'name': 'lymphocyte', 'id': 3, 'trainId': 2},
    {'color': [255, 255, 0], 'name': 'plasma', 'id': 4, 'trainId': 3},
    {'color': [255, 0, 255], 'name': 'eosinophil', 'id': 5, 'trainId': 4},
    {'color': [0, 255, 255], 'name': 'connective', 'id': 6, 'trainId': 5},
]

CLASSES = ('neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective')
PALETTE = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
# yapf:enable


def load_conic(split_path, split_list):

    dataset_dicts = []
    for img_id in split_list:
        dataset_dict = {}
        dataset_dict['file_name'] = osp.join(split_path, f'{img_id}.png')
        dataset_dict['sem_ann_file_name'] = osp.join(split_path, f'{img_id}_semantic.png')
        dataset_dict['inst_ann_file_name'] = osp.join(split_path, f'{img_id}_instance.npy')

        dataset_dicts.append(dataset_dict)

    return dataset_dicts


def register_conic(root):
    root = osp.join(root, "conic")

    for name, split in [("train", "train.txt"), ("val", "val.txt")]:
        split_path = osp.join(root, name)
        split_txt = osp.join(root, split)
        split_list = []
        for line in open(split_txt, 'r').readlines():
            img_id = line.strip()
            split_list.append(img_id)

        name = f"conic_{name}"

        DatasetCatalog.register(name, lambda x=split_path, y=split_list: load_conic(x, y))

        MetadataCatalog.get(name).set(
            cat_info=CONIC_CATEGORIES,
            classes=CLASSES,
            palette=PALETTE,
            path=split_path,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_conic(_root)
