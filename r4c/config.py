def add_local_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # data config
    cfg.INPUT.TEST_DATASET_MAPPER_NAME = None
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = None
    # set input base shape (yolo style)
    cfg.INPUT.INPUT_BASE_SHAPE = 416
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MIN_SIZE_TRAIN = (800, )
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    # CLASSES
    cfg.INPUT.NUM_CLASSES = 1

    # model config
    # the shape / the output shape
    cfg.MODEL.OUTPUT_STRIDE = 1
    # backbone pretrain model
    cfg.MODEL.BACKBONE_WEIGHTS = 'pretrained/yolov3.pth'
    # backbone name
    cfg.MODEL.BACKBONE_NAME = 'r50'

    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAM"
    cfg.SOLVER.EPS = 1e-8
    cfg.SOLVER.BACKBONE_LR_RATIO = 0.00005
