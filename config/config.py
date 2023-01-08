from yacs.config import CfgNode as CN
import pprint

_C = CN()
_C.GPUS = (0,)
_C.LOG_DIR = './log'
_C.DATA_DIR = '../dataset'
_C.OUTPUT_DIR = './result'
_C.WORKERS = 8
_C.BATCH_SIZE = 32

_C.DATASET = CN()
_C.DATASET.TRAIN_SET = CN()
_C.DATASET.TRAIN_SET.COMPRESSION = 'c23'
_C.DATASET.TRAIN_SET.BLENDED_IMAGES = CN()
_C.DATASET.TRAIN_SET.BLENDED_IMAGES.CANDIDATES = 100
_C.DATASET.TRAIN_SET.BLENDED_IMAGES.SUBSET_SIZE = 200
_C.DATASET.TRAIN_SET.BLENDED_IMAGES.USE = True
_C.DATASET.TRAIN_SET.FF_USE = []
_C.DATASET.VAL_SET = CN()
_C.DATASET.VAL_SET.COMPRESSION = 'c23'
_C.DATASET.VAL_SET.FF_USE = ['Deepfakes']
_C.DATASET.TEST_SET = CN()
_C.DATASET.TEST_SET.COMPRESSION = 'c23'
_C.DATASET.TEST_SET.FF_USE = ['Deepfakes']

_C.TRAIN = CN()
_C.TRAIN.LAMBDA = 100
_C.TRAIN.TOTAL_EPOCHS = 30
_C.TRAIN.WARM_UP_EPOCHS = 10
_C.TRAIN.LR = 0.0002
_C.TRAIN.OPTIMIZER = 'Adam'
_C.TRAIN.LR_STEP = [15, 25]
_C.TRAIN.GAMMA = 0.5


def update_config(config, config_file):
    config.defrost()
    config.merge_from_file(config_file)
    config.freeze()


if __name__ == '__main__':
    update_config(_C, '../experiments/default.yaml')

    pprint.pprint(_C)

