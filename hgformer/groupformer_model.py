# This version is used to implement loss on multiple stages
# Copyright (c) Facebook, Inc. and its affiliates.
import imp
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterionSpix
from .modeling.matcher import HungarianMatcher
from skimage import color

@META_ARCH_REGISTRY.register()
class GroupFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        dynamic_mean_std: bool,
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        cluster_softmax: bool,
        pred_stage: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.dynamic_mean_std = dynamic_mean_std
        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.cluster_softmax = cluster_softmax
        self.pred_stage = pred_stage

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        spix_sem_weight = cfg.MODEL.MASK_FORMER.SPIX_MASK_WEIGHT
        spix_color_weight = cfg.MODEL.MASK_FORMER.SPIX_COLOR_WEIGHT
        spix_pixel_cls_weight = cfg.MODEL.MASK_FORMER.SPIX_CLASS_WEIGHT
        pixel_cls_loss_weight = cfg.MODEL.MASK_FORMER.PIXEL_CLASS_WEIGHT
        contrastive_loss_weight = cfg.MODEL.MASK_FORMER.CONTRASTIVE_WEIGH
        spix_pixel_cls_loss_weight = cfg.MODEL.MASK_FORMER.SPIX_PIXEL_CLS_WEIGH
        stage_weight = cfg.MODEL.MASK_FORMER.STAGE_WEIGHTS
        spix_mask_stage2_weight = cfg.MODEL.MASK_FORMER.SPIX_MASK_STAGE2
        num_group_tokens = cfg.MODEL.SEM_SEG_HEAD.NUM_GROUP_TOKENS
        assert (len(num_group_tokens) - 1) == len(stage_weight)
        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight} # stage3

        spix_weight_dict = {"loss_sem": spix_sem_weight, "loss_pixel_cls": spix_pixel_cls_weight,
                            "loss_coord": spix_sem_weight * 0.01, "loss_color": spix_color_weight}

        pixel_weight_dict = {"pixel_cls_loss": pixel_cls_loss_weight}

        contrastive_weight_dict = {"contrastive_loss": contrastive_loss_weight}

        spix_pixel_cls_loss_weight_dict = {"spix_pixel_cls_loss": spix_pixel_cls_loss_weight}

        weight_dict_spix = {"spix_pixel_cls_loss": spix_pixel_cls_loss_weight, "contrastive_loss": contrastive_loss_weight}

        if deep_supervision:
            num_stages = len(num_group_tokens)
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS

            aux_weight_dict = {}

            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in spix_weight_dict.items()})
            # import ipdb; ipdb.set_trace()
            aux_weight_dict["loss_sem_1"] = aux_weight_dict["loss_sem_1"] * spix_mask_stage2_weight

            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})

            spix_layers = cfg.MODEL.MASK_FORMER.SPIX_SELF_ATTEN_LAYERS
            aux_weight_dict_spix = {}
            for i in range(spix_layers):
                aux_weight_dict_spix.update({k + f"_{i}": v for k, v in weight_dict_spix.items()})

            weight_dict.update(aux_weight_dict)
            weight_dict.update(pixel_weight_dict)
            weight_dict.update(contrastive_weight_dict)
            weight_dict.update(spix_pixel_cls_loss_weight_dict)
            weight_dict.update(aux_weight_dict_spix)

        losses = ["labels", "masks"]
        # import ipdb; ipdb.set_trace()
        criterion = SetCriterionSpix(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            reconstuct_loss=cfg.MODEL.MASK_FORMER.RECONSTRUCT_LOSS,
            reconstruct_coord=cfg.MODEL.MASK_FORMER.RECONSTRUCT_COORD,
            reconstruct_color=cfg.MODEL.MASK_FORMER.RECONSTRUCT_COLOR,
            contrastive_loss=cfg.MODEL.MASK_FORMER.CONTRASTIVE_LOSS,
            contrastive_tau=cfg.MODEL.MASK_FORMER.CONTRASTIVE_TAU,
            high_threshold=cfg.MODEL.MASK_FORMER.HIGH_THRESHOLD,
            low_threshold=cfg.MODEL.MASK_FORMER.LOW_THRESHOLD,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dynamic_mean_std": cfg.MODEL.DYNAMIC_MEN_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "cluster_softmax": cfg.TEST.CLUSTER_SOFTMAX,
            "pred_stage": cfg.TEST.PRED_STAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        # import ipdb; ipdb.set_trace()

        if not self.dynamic_mean_std:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
        else:
            image_list = []
            for x in images:
                current_pixel_mean = torch.Tensor([x[0].float().mean(), x[1].float().mean(), x[2].float().mean()]).view(-1, 1, 1).to(x.device)
                current_pixel_var = torch.Tensor([x[0].float().var(), x[1].float().var(), x[2].float().var()]).view(-1, 1, 1).to(x.device)
                current_image = (x - current_pixel_mean) / (current_pixel_var + 1.e-6)**.5
                image_list.append(current_image)
            images = ImageList.from_tensors(image_list, self.size_divisibility)

        if self.training:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)
            # mask classification target
            if "instances" in batched_inputs[0]:
                # "height": 1024, "width": 2048 (original shape)
                # "image".shape: [3, 512, 1024] (resized shape)
                # "sem_seg".shape: [512, 1024] (same as resized shape)
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_sem_segs = [x["sem_seg"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, gt_sem_segs, images)
            else:
                targets = None
            # bipartite matching-based loss
            # outputs['pred_masks'].shape: [2, 64, 64, 128]
            # target[0]['masks'].shape [9, 512, 1024]
            losses = self.criterion(outputs, targets)
            # import ipdb; ipdb.set_trace()
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            if self.cluster_softmax:

                pixel_level_logits = outputs["pixel_level_logits"]
                # import ipdb; ipdb.set_trace()
                pixel_level_logits = F.interpolate(pixel_level_logits,
                                                   size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                                                   mode="bilinear",
                                                   align_corners=False,)
                processed_results_pixel = []
                for pixel_level_logit, input_per_image, image_size in zip(
                        pixel_level_logits, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results_pixel.append({})
                    # import ipdb; ipdb.set_trace()
                    sem_seg = pixel_level_logit.softmax(0)
                    sem_seg = retry_if_cuda_oom(sem_seg_postprocess)(sem_seg, image_size, height, width)
                    processed_results_pixel[-1]["sem_seg"] = sem_seg

                if "predicitons_class_spix_pixel_cls" in outputs:
                    predicitons_class_spix_pixel_cls = outputs["predicitons_class_spix_pixel_cls"]
                    processed_results_spix_pixel_results = []
                    for i in range(len(predicitons_class_spix_pixel_cls)):
                        seg_logits = predicitons_class_spix_pixel_cls[i]
                        seg_logits = F.interpolate(seg_logits,
                                                   size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                                                   mode="bilinear",
                                                   align_corners=False, )
                        tmp_processed_results = []
                        for seg_logit, input_per_image, image_size in zip(
                                seg_logits, batched_inputs, images.image_sizes):
                            height = input_per_image.get("height", image_size[0])
                            width = input_per_image.get("width", image_size[1])
                            tmp_processed_results.append({})
                            sem_seg = seg_logit.softmax(0)
                            sem_seg = retry_if_cuda_oom(sem_seg_postprocess)(sem_seg, image_size, height, width)
                            tmp_processed_results[-1]["sem_seg"] = sem_seg
                        processed_results_spix_pixel_results.append(tmp_processed_results)

            del outputs
            # print('------------------------------------------------mask inference-----------------------------------------------------------------------------------')
            processed_results = []
            # mask_cls_results: [1, 100, 20], mask_pred_results: [1, 100, 1024, 1824]
            # height: 1080, width: 1920, image_size: [1024, 1820]
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                # self.sem_seg_postprocess_before_inference: False
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

            if self.cluster_softmax:
                if self.pred_stage == "spix_all_stage_exclude0125":
                    # import ipdb; ipdb.set_trace()
                    for i in range(len(processed_results)):
                        for j in range(3, len(processed_results_spix_pixel_results) - 1):
                            processed_results[i]["sem_seg"] = processed_results[i]["sem_seg"] + \
                                                                  + processed_results_spix_pixel_results[j][i]["sem_seg"].to(processed_results[i]["sem_seg"].device)
                        processed_results[i]["sem_seg"] = processed_results[i]["sem_seg"] + processed_results_pixel[i]["sem_seg"].to(processed_results[i]["sem_seg"].device)
                elif self.pred_stage == "spix_pixelexclude0125+stage3":
                    # import ipdb; ipdb.set_trace()
                    for i in range(len(processed_results)):
                        for j in range(3, len(processed_results_spix_pixel_results) - 1):
                            processed_results[i]["sem_seg"] = processed_results[i]["sem_seg"] + \
                                                                  + processed_results_spix_pixel_results[j][i]["sem_seg"].to(processed_results[i]["sem_seg"].device)

            # TODO: if test on ACDC test, write prediction pngs here

            return processed_results

    def prepare_targets(self, gt_instances, gt_sem_segs, images):
        # just used for padding. If the images have different sizes, padding is required
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for i in range(len(gt_instances)):
            # pad gt
            instance = gt_instances[i]
            sem_seg = gt_sem_segs[i]
            image = images.tensor[i]

            gt_masks = instance.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            # TODO: how to padd the sem_seg, refer to deeplab
            padded_masks_sem_seg = torch.ones((h_pad, w_pad), dtype=sem_seg.dtype, device=sem_seg.device)
            padded_masks_sem_seg = padded_masks_sem_seg * 255 # 255 means ignored pixels
            # import ipdb; ipdb.set_trace()
            padded_masks_sem_seg[: sem_seg.shape[0], :sem_seg.shape[1]] = sem_seg
            new_targets.append(
                {
                    "labels": instance.gt_classes,
                    "masks": padded_masks,
                    "sem_seg": padded_masks_sem_seg,
                    "image": image,
                }
            )

        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
