from easydict import EasyDict as edict

__C = edict()
cfg = __C

# general setting
__C.BATCH_SIZE = 64
__C.NAME = "GAWWN"
__C.ROOT_PATH = "/data/cl/CUB_200_2011"
#__C.ROOT_PATH = "/home/joshua/CUB"
__C.NEPOCHS = 1000
__C.CHECKPOINT_PERIOD = 100
__C.LR = 0.0002
__C.BETAS = (0.5, 0.999)
__C.LR_DECAY = 0.5
__C.DECAY_PERIOD = 100
__C.OUTPUT_DIR = "output"

# Tree Options
__C.TREE = edict()
__C.TREE.BRANCH_NUM = 1
__C.TREE.BASE_SIZE = 128

# Model options
__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.TXT_EMBEDDING_DIM = 1024
__C.TEXT.TXT_FEATURE_DIM = 128

__C.KEYPOINT = edict()
__C.KEYPOINT.NUM_ELT = 15
__C.KEYPOINT.DIM = 16

__C.IMAGE = edict()
__C.IMAGE.LOADSIZE = 150
__C.IMAGE.FINESIZE = 128

__C.GAN = edict()
__C.GAN.CLS_WEIGHT = 0.5
__C.GAN.FAKE_SCORE = 0.5
__C.GAN.Z_DIM = 100
__C.GAN.LOC_DIM = 16 
__C.GAN.NGF = 128   # generator filters in first conv layer
__C.GAN.NDF = 64    # discrimnator filters in first conv layer
