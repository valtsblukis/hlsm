from typing import Dict

import torch
import torch.nn as nn

from lgp.abcd.model import LearnableModel
from lgp.utils.viz import show_image
from lgp.flags import GLOBAL_VIZ

import lgp.env.alfred.segmentation_definitions as segdef

from lgp.models.alfred.hlsm.unets.unet_5 import UNet5

from lgp.ops.depth_estimate import DepthEstimate


class AlfredSegmentationAndDepthModel(LearnableModel):
    TRAINFOR_SEG = "segmentation"
    TRAINFOR_DEPTH = "depth"
    TRAINFOR_BOTH = "both"

    """
    Given a current state s_t, proposes an action distribution that makes sense.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hidden_dim = 128
        self.semantic_channels = segdef.get_num_objects()

        self.params = hparams.get("perception_model")

        # Training hyperparams
        self.train_for = self.params.get("train_for", self.TRAINFOR_BOTH)

        # Inference hyperparams
        self.depth_t_beta = self.params.get("depth_t_beta", 0.5)
        self.seg_t_beta = self.params.get("seg_t_beta", 1.0)

        # Model hyperparams
        self.distr_depth = self.params.get("distributional_depth", True)
        self.depth_bins = self.params.get("depth_bins", 50)
        self.max_depth_m = self.params.get("max_depth", 5.0)

        assert self.train_for in [self.TRAINFOR_SEG, self.TRAINFOR_DEPTH, self.TRAINFOR_BOTH, None]
        print(f"Training perception model for: {self.train_for}")

        self.net = UNet5(self.distr_depth, self.depth_bins)

        self.iter = nn.Parameter(torch.zeros([1], dtype=torch.double), requires_grad=False)

        self.nllloss = nn.NLLLoss(reduce=True, size_average=True)
        self.celoss = nn.CrossEntropyLoss(reduce=True, size_average=True)
        self.mseloss = nn.MSELoss(reduce=True, size_average=True)
        self.act = nn.LeakyReLU()

    def predict(self, rgb_image):
        with torch.no_grad():
            if self.distr_depth:
                seg_pred, depth_pred = self.forward_model(rgb_image)
                seg_pred = torch.exp(seg_pred * self.seg_t_beta)
                depth_pred = torch.exp(depth_pred * self.depth_t_beta)
                depth_pred = depth_pred / (depth_pred.sum(dim=1, keepdim=True))

                depth_pred = DepthEstimate(depth_pred, self.depth_bins, self.max_depth_m)

                # Filter segmentations
                good_seg_mask = seg_pred > 0.3
                seg_pred = seg_pred * good_seg_mask
                seg_pred = seg_pred / (seg_pred.sum(dim=1, keepdims=True) + 1e-10)
            else:
                seg_pred, depth_pred = self.forward_model(rgb_image)
                seg_pred = torch.exp(seg_pred)

                good_seg_mask = seg_pred > 0.3
                good_depth_mask = (seg_pred > 0.5).sum(dim=1, keepdims=True) * (depth_pred > 0.9)
                seg_pred = seg_pred * good_seg_mask
                seg_pred = seg_pred / (seg_pred.sum(dim=1, keepdims=True) + 1e-10)
                depth_pred = depth_pred * good_depth_mask

        return seg_pred, depth_pred

    def forward_model(self, rgb_image: torch.tensor):
        return self.net(rgb_image)

    def get_name(self) -> str:
        return "alfred_segmentation_and_depth_model"

    def loss(self, batch: Dict):
        # This is now forward
        return self.forward(batch)

    def forward(self, batch: Dict):
        # Inputs
        observations = batch["observations"]
        rgb_image = observations.rgb_image.float()
        seg_gt = observations.semantic_image.float().clone()
        depth_gt = observations.depth_image.float()
        # Switch to a one-hot segmentation representation
        observations.uncompress()
        seg_gt_oh = observations.semantic_image.float()

        b, c, h, w = seg_gt.shape

        # Model forward pass
        seg_pred, depth_pred = self.forward_model(rgb_image)

        # Depth inference and error signal computation
        c = seg_pred.shape[1]
        seg_flat_pred = seg_pred.permute((0, 2, 3, 1)).reshape([b * h * w, c])
        seg_flat_gt = seg_gt.permute((0, 2, 3, 1)).reshape([b * h * w]).long()

        seg_loss = self.nllloss(seg_flat_pred, seg_flat_gt)

        if self.distr_depth:
            depth_flat_pred = depth_pred.permute((0, 2, 3, 1)).reshape([b * h * w, self.depth_bins])
            depth_flat_gt = depth_gt.permute((0, 2, 3, 1)).reshape([b * h * w])
            depth_flat_gt = ((depth_flat_gt / self.max_depth_m).clamp(0, 0.999) * self.depth_bins).long()
            depth_loss = self.nllloss(depth_flat_pred, depth_flat_gt)

            depth_pred_mean = (torch.arange(0, self.depth_bins, 1, device=depth_pred.device)[None, :, None, None] * torch.exp(depth_pred)).sum(dim=1)
            depth_mae = (depth_pred_mean.view([-1]) - depth_flat_gt).abs().float().mean() * (self.max_depth_m / self.depth_bins)
        else:
            depth_flat_pred = depth_pred.reshape([b, h * w])
            depth_flat_gt = depth_gt.reshape([b, h * w])
            depth_loss = self.mseloss(depth_flat_pred, depth_flat_gt)
            depth_mae = (depth_flat_pred - depth_flat_gt).abs().mean()

        seg_pred_distr = torch.exp(seg_pred)

        # Loss computation
        if self.train_for is None:
            raise ValueError("train_for hyperparameter not set")
        if self.train_for == self.TRAINFOR_DEPTH:
            loss = depth_loss
        elif self.train_for == self.TRAINFOR_SEG:
            loss = seg_loss
        elif self.train_for == self.TRAINFOR_BOTH:
            loss = seg_loss + depth_loss
        else:
            raise ValueError(f"Unrecognized train_for setting: {self.train_for}")

        # Visualization code (removing this doesn't affect functionality)
        if GLOBAL_VIZ and self.distr_depth:
            self._real_time_visualization(seg_pred_distr, seg_gt_oh, rgb_image, depth_pred, depth_pred_mean, depth_gt)

        # Outputs
        metrics = {}
        metrics["loss"] = loss.item()
        metrics["seg_loss"] = seg_loss.item()
        metrics["depth_loss"] = depth_loss.item()
        metrics["depth_mae"] = depth_mae.item()

        self.iter += 1

        return loss, metrics

    def _real_time_visualization(self, seg_pred_distr, seg_gt_oh, rgb_image, depth_pred, depth_pred_mean, depth_gt):
        def map_colors_for_viz(cdist):
            # Red - 0,  Blue - 1,    bluegreen - 2,    Yellow - 3
            colors = segdef.get_class_color_vector().to(cdist.device).float() / 255.0
            # B x 4 x 3 x H x W
            colors = colors[None, :, :, None, None]
            cdist = cdist[:, :, None, :, :]
            pdist_c = (cdist * colors).sum(dim=1).clamp(0, 1)
            return pdist_c

        if self.iter.item() % 10 == 0:
            with torch.no_grad():
                seg_pred_viz = map_colors_for_viz(seg_pred_distr)[0].permute((1, 2, 0)).detach().cpu().numpy()
                seg_gt_viz = map_colors_for_viz(seg_gt_oh)[0].permute((1, 2, 0)).detach().cpu().numpy()
                rgb_viz = rgb_image[0].permute((1, 2, 0)).detach().cpu().numpy()

                show_image(rgb_viz, "rgb", scale=1, waitkey=1)
                show_image(seg_pred_viz, "seg_pred", scale=1, waitkey=1)
                show_image(seg_gt_viz, "seg_gt", scale=1, waitkey=1)

                if self.distr_depth:
                    depth_pred_amax_viz = depth_pred[0].argmax(0).detach().cpu().numpy()
                    depth_pred_mean_viz = depth_pred_mean[0].detach().cpu().numpy()
                    depth_pred_std = depth_pred[0].std(0).detach().cpu().numpy()
                    depth_gt_viz = depth_gt[0].permute((1, 2, 0)).detach().cpu().numpy()
                    show_image(depth_pred_amax_viz, "depth_pred_amax", scale=1, waitkey=1)
                    show_image(depth_pred_mean_viz, "depth_pred_mean_viz", scale=1, waitkey=1)
                    show_image(depth_pred_std, "depth_pred_std", scale=1, waitkey=1)
                    show_image(depth_gt_viz, "depth_gt", scale=1, waitkey=1)
                else:
                    depth_pred_viz = depth_pred[0].permute((1, 2, 0)).detach().cpu().numpy()
                    depth_gt_viz = depth_gt[0].permute((1, 2, 0)).detach().cpu().numpy()
                    show_image(depth_pred_viz, "depth_pred", scale=1, waitkey=1)
                    show_image(depth_gt_viz, "depth_gt", scale=1, waitkey=1)


import lgp.model_registry
lgp.model_registry.register_model("alfred_perception_model", AlfredSegmentationAndDepthModel)
