# HGFormer Model Zoo and Baselines

#### Detectron2 ImageNet Pretrained Models

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. 

To prepare the backbones pre-trained on ImageNet classification, please following [tools/README.md](./tools/README.md)

#### License

All models available for download through this document are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

## Cityscapes -> ACDC
|    Method   |  Backbone |  Fog  | Night |  Rain |  Snow |  All  | Download |
|:-----------:|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:--------:|
| Mask2former | Swin-Tiny | 54.06 | 38.11 | 59.54 | 55.76 | 53.65 |   [model](https://drive.google.com/drive/folders/1eL38sFGdUNV8o9EbFsheurHjdm-CNg5K?usp=sharing)  |
|   HGFormer  | Swin-Tiny | 59.82 | 41.88 | 60.92 | 60.82 | 56.95 |   [model](https://drive.google.com/drive/folders/1Rq1PnaYTFACpZX_-oXTq7laCa0zwbfFR?usp=drive_link)  |


## Cityscapes -> Cityscapes-C (level 5)
|    Method   | Backbone  | Average | Motion | Defoc | Glass | Gauss | Gauss | Impul |  Shot | Speck | Bright | Contr | Satur |  JPEG |  Snow | Spatt |  Fog  | Frost | Download |
|:-----------:|:-----------:|:---------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:--------:|
| Mask2former | Swin-Tiny |  41.68  |  51.61 | 51.52 | 39.69 | 46.71 |  6.89 |  7.68 | 12.75 | 44.10 |  72.71 | 58.60 | 69.14 | 22.86 | 26.10 | 58.35 | 67.12 | 31.11 |   [model](https://drive.google.com/drive/folders/1eL38sFGdUNV8o9EbFsheurHjdm-CNg5K?usp=sharing)  |
|   HGFormer  | Swin-Tiny |  43.81  |  52.51 | 53.03 | 39.02 | 47.93 | 16.45 | 16.03 | 20.55 | 48.44 |  74.51 | 57.14 | 70.53 | 27.32 | 25.66 | 59.19 | 66.49 | 26.11 |   [model](https://drive.google.com/drive/folders/1Rq1PnaYTFACpZX_-oXTq7laCa0zwbfFR?usp=drive_link)  |
## Cityscapes -> Others
|    Method   | Backbone  | Mapillary |  BDD  |  GTA  | Synthia | Average | Download |
|:-----------:|:---------:|:---------:|:-----:|:-----:|:-------:|:--------:|:--------:|
| Mask2former | Swin-Tiny |   65.28   | 49.87 | 51.38 |  34.76  | 50.32   |  [model](https://drive.google.com/drive/folders/1eL38sFGdUNV8o9EbFsheurHjdm-CNg5K?usp=sharing)  |
|   HGFormer  | Swin-Tiny |   67.22   | 52.69 | 51.94 |  32.98  |  51.21  |[model](https://drive.google.com/drive/folders/1Rq1PnaYTFACpZX_-oXTq7laCa0zwbfFR?usp=drive_link)  |
## Mapillary -> Others

|    Method   |  Backbone |  GTA  | Synthia | Cityscapes |  BDD  | Average | Download |
|:-----------:|:---------:|:-----:|:-------:|:----------:|:-----:|:-------:|:--------:|
| Mask2former | Swin-Tiny | 57.81 |   40.14 |      68.23 | 59.05 |  56.31  |   [model](https://drive.google.com/drive/folders/1xqvAcQZs2NZhUD5dG2KGPmYBnlkH4u-s?usp=drive_link)  |
|   HGFormer  | Swin-Tiny | 60.79 |   39.15 |      69.28 | 62.22 |  57.86  |   [model](https://drive.google.com/drive/folders/1XJgHBKT7J-_Gzqgzo3EiX0wAnjXMNCGG?usp=drive_link)  |

## Disclaimer
The numbers differ slightly from the results reported in the paper because we presented an average of three runs in the paper.