# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.COLOR_AUG_MIX = 'partial'
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.DEEP_MASK_SUPERVISION = False
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.SPIX_MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.SPIX_COLOR_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.SPIX_CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.PIXEL_CLASS_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.REGION_PROXY_CLS_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.CONTRASTIVE_WEIGH = 2.0
    cfg.MODEL.MASK_FORMER.CONTRASTIVE_LOSS = False
    # cfg.MODEL.MASK_FORMER.EDGE_DISTANCES = [1, 2, 4, 8]
    cfg.MODEL.MASK_FORMER.HIGH_THRESHOLD = 0.3
    cfg.MODEL.MASK_FORMER.LOW_THRESHOLD = 0.05
    cfg.MODEL.MASK_FORMER.RETURN_ITERATION = False
    cfg.MODEL.MASK_FORMER.OBLIQUE_DISTANCES = [1, 2, 4, 8]
    # cfg.MODEL.MASK_FORMER.BYOL_WEIGH = 2.0
    # cfg.MODEL.MASK_FORMER.EDGE_WEIGH = 2.0
    # cfg.MODEL.MASK_FORMER.PSEUDO_EDGE_WEIGH = 2.0
    cfg.MODEL.MASK_FORMER.SPIX_PIXEL_CLS_WEIGH = 2.0
    # cfg.MODEL.MASK_FORMER.BYOL_LOSS = False
    # cfg.MODEL.MASK_FORMER.EDGE_LOSS = False
    cfg.MODEL.MASK_FORMER.CONTRASTIVE_TAU = 0.3
    cfg.MODEL.MASK_FORMER.COMPUTE_RAMA = False
    cfg.MODEL.MASK_FORMER.RECONSTRUCT_LOSS = False
    cfg.MODEL.MASK_FORMER.RECONSTRUCT_COLOR = False
    cfg.MODEL.MASK_FORMER.RECONSTRUCT_COORD = False
    cfg.MODEL.MASK_FORMER.STAGE_WEIGHTS = [1.0, 1.0]
    cfg.MODEL.MASK_FORMER.SPIX_MASK_STAGE2 = 1.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.SPIX_SELF_ATTEN_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.CONTRASTIVE_DIM = 128
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    # cfg.TEST.MODE = "whole" # "whole" or "slide"
    # cfg.TEST.STRIDE = (300, 768)
    # cfg.TEST.CROP_SIZE = (512, 1024)
    cfg.TEST.CLUSTER_SOFTMAX = False
    cfg.TEST.PRED_STAGE = "all"

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    cfg.MODEL.MASK_FORMER.GZERO_CALIBRATE = -1.0
    cfg.MODEL.MASK_FORMER.ENSEMBLING = False
    cfg.MODEL.MASK_FORMER.ENSEMBLING_ALL_CLS = False

    # vis
    cfg.MODEL.MASK_FORMER.VIS = False
    cfg.MODEL.MASK_FORMER.QUERY_SHAPE = [8, 16] # h, w
    cfg.MODEL.MASK_FORMER.ENSEMBLING_START = 1

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"
    # gzero calibrate
    cfg.MODEL.SEM_SEG_HEAD.GZERO_CALIBRATE = -1.0

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # pvt backbone
    cfg.MODEL.PVTV2 = CN()
    cfg.MODEL.PVTV2.PATCH_SIZE = 4
    cfg.MODEL.PVTV2.IN_CHANS = 3
    cfg.MODEL.PVTV2.EMBED_DIMS = [32, 64, 160, 256]
    cfg.MODEL.PVTV2.NUM_HEADS = [1, 2, 5, 8]
    cfg.MODEL.PVTV2.MLP_RATIO = [8, 8, 4, 4]
    cfg.MODEL.PVTV2.QKV_BIAS = True
    cfg.MODEL.PVTV2.DROP_RATE = 0.0
    cfg.MODEL.PVTV2.DROP_PATH_RATE = 0.
    cfg.MODEL.PVTV2.QK_SCALE = None
    cfg.MODEL.PVTV2.DEPTHS = [2, 2, 2, 2]
    cfg.MODEL.PVTV2.SR_RATIOS = [8, 4, 2, 1]
    cfg.MODEL.PVTV2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]


    cfg.MODEL.SEM_SEG_HEAD.MASKATTENTIONPOOL = False
    cfg.MODEL.SEM_SEG_HEAD.TEMPERATURE = 0.01
    cfg.MODEL.SEM_SEG_HEAD.GAT_NUM_LAYERS = 2
    cfg.MODEL.SEM_SEG_HEAD.DOWNSAMPLE_RATE = 4
    # cfg.MODEL.CRITERION = "spix" # default

    # self training config
    cfg.MODEL.PSEUDO_LABEL = False
    cfg.MODEL.PSEUDO_WEIGHT = 1.0
    cfg.MODEL.PSEUDO_THR = -1.


    cfg.MODEL.DYNAMIC_MEN_STD = False
    # cfg.MODEL.LAB_INPUT = False

    # NOTE: maskformer2 extra conffigs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # params for groupformer
    cfg.MODEL.SEM_SEG_HEAD.NUM_GROUP_TOKENS = [256, 128, 64]
    cfg.MODEL.SEM_SEG_HEAD.NUM_OUTPUT_GROUPS = [256, 128, 64]
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = [8, 8, 8]
    cfg.MODEL.SEM_SEG_HEAD.SPIX_RES = [32, 32]
    cfg.MODEL.SEM_SEG_HEAD.MASK_POOL_STYLE = "attn_pool"
    cfg.MODEL.SEM_SEG_HEAD.TAU = 0.07

    cfg.MODEL.OUT_SUBMISSION_FORMAT = False

    cfg.MODEL.SEM_SEG_HEAD.SPIX_SELF_ATTEN = True
    cfg.MODEL.SEM_SEG_HEAD.SPIX_FFN = True
