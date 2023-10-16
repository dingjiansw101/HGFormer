# Getting Started with HGFormer

This document provides a brief intro of the usage of HGFormer.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

## Evaluation with Pre-trained Models

Download [models](https://drive.google.com/drive/folders/1fUWaIhXtSxHLdTFxnuOSldLUe_ferauh?usp=drive_link).

### Cityscapes -> ACDC

```
python demo/inference.py --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
--input datasets/acdc/rgb_anon/all/test --output path_to_output \
--opts MODEL.WEIGHTS path_to_checkpoint
```
After running the command, you will find the results on ```path_to_output```. Then you can follow the instructions on [ACDC evaluation server](https://acdc.vision.ee.ethz.ch/login?target=%2Fsubmit) to get your scores.
You can replace ```all``` with a specific type ```fog, snow, night, rain```, if you want to evaluate on a specific type

### Cityscapes -> Cityscapes-c

```
python test_city_c_level5.py --num-gpus 8 --config-file configs/city_c/hgformer_swin_tiny_bs16_20k.yaml \
 --eval-only MODEL.WEIGHTS path_to_checkpoint OUTPUT_DIR path_to_output
```

### Cityscapes -> Others

```
python plain_train_net.py --num-gpus 8 --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
--eval-only MODEL.WEIGHTS path_to_checkpoint OUTPUT_DIR path_to_output
```

### Mapillary -> Others

```
python plain_train_net.py --num-gpus 8 --config-file configs/mapillary/hgformer_swin_tiny_bs16_20k_mapillary.yaml \
--eval-only MODEL.WEIGHTS path_to_checkpoint OUTPUT_DIR path_to_output
```

## Training in Command Line


To train a model, first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md), then prepare the models pre-trained on ImageNet classificaiton following [tools/README.md](./tools/README.md). Finally run:
```
python plain_train_net.py --num-gpus 8 \
  --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml OUTPUT_DIR path_to_output
```

The configs are made for 8-GPU training.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
To train on 1 GPU, you need to figure out learning rate and batch size by yourself:
```
python plain_train_net.py \
  --config-file configs/cityscapes/hgformer_swin_tiny_bs16_20k.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
```
