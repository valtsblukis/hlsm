from typing import Dict

import numpy as np
import math

import torch
import torch.nn as nn

from lgp.abcd.model import LearnableModel
from lgp.ops.spatial_distr import multidim_logsoftmax

from lgp.utils.viz import show_image

from lgp.env.alfred.alfred_subgoal import AlfredSubgoal

from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from lgp.models.alfred.hlsm.unets.lingunet_3 import Lingunet3


MODEL_ROTATION = True


class HlsmNavigationModel(LearnableModel):

    """
    Given a current state s_t, proposes an action distribution that makes sense.
    """
    def __init__(self, hparams = None):
        super().__init__()
        self.hidden_dim = 128
        self.feature_2d_dim = AlfredSpatialStateRepr.get_2d_feature_dim()
        self.action_2d_dim = 2

        out_channels = 6 if MODEL_ROTATION else 1
        self.lingunet = Lingunet3(
            in_channels=self.feature_2d_dim + self.action_2d_dim,
            context_size=self.hidden_dim,
            out_channels=out_channels)
        self.act_type_emb = nn.Embedding(AlfredSubgoal.get_action_type_space_dim(), self.hidden_dim)
        self.act_arg_emb = nn.Embedding(AlfredSubgoal.get_action_arg_space_dim() + 1, self.hidden_dim)
        self.act_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.iter = nn.Parameter(torch.zeros([1]), requires_grad=False)

        self.nllloss = nn.NLLLoss(reduce=True, size_average=True)
        self.act = nn.LeakyReLU()

    def mle(self,
            state: AlfredSpatialStateRepr,
            object_id: torch.tensor):
        return ...

    def forward_model(self,
                      features_2d: torch.tensor,
                      subgoal_arg_features: torch.tensor,
                      subgoal_tensors: torch.tensor):
        # Inputs
        act_types = subgoal_tensors[:, 0]
        act_args = subgoal_tensors[:, 1] + 1
        lingin = torch.cat([features_2d, subgoal_arg_features], dim=1)

        # Action representation
        type_emb = self.act(self.act_type_emb(act_types))
        arg_emb = self.act(self.act_arg_emb(act_args))
        act_emb = self.act_linear(type_emb + arg_emb)

        # Goal prediction
        out = self.lingunet(lingin, act_emb)

        # Unpacking results
        pos_logits = out[:, 0:1, :, :]
        yaw_class = out[:, 1:5, :, :]
        pitch_reg = out[:, 5:6, :, :]

        # Activations
        pos_log_distr = multidim_logsoftmax(pos_logits, dims=(2, 3))  # P(x, y)
        yaw_log_distr = multidim_logsoftmax(yaw_class, dims=(1,))     # P(yaw | x, y)
        pitch_pred = torch.tanh(pitch_reg) * math.pi                  # P(pitch | yaw, x, y)
        # Pitch is always between -pi and pi

        return pos_log_distr, yaw_log_distr, pitch_pred

    def get_name(self) -> str:
        return "alfred_spatial_exploration_model"

    def success(self, pred_logits, class_indices):
        # TODO: Measure distance between argmax ground truth and predicted
        return ...

    def collect_metrics(self, location_logdistr, gt_location_distr):
        metrics = {
        }
        return metrics

    def loss(self, batch: Dict):
        # This is now forward
        return self.forward(batch)

    def forward(self, batch: Dict):
        # TODO: Add dataset collate function that centers features around the agent position
        state_images = batch["state_images"]
        features_2d = batch["features_2d"]
        subgoals = batch["subgoals"]
        subgoal_arg_features = batch["subgoal_args"]
        gt_pos_yaw_prob = batch["nav_goal_images"]
        gt_pitch = batch["nav_goal_pitch_images"]

        #print(gt_pitch.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist())
        b, c, h, w = features_2d.shape
        pos_pred_log_distr, yaw_log_distr, pitch_prediction = self.forward_model(features_2d, subgoal_arg_features, subgoals)

        # Loss for predicting the goal position
        gt_pos_prob = gt_pos_yaw_prob.sum(dim=1, keepdim=True)
        flat_gt_pos_prob = gt_pos_prob.view((b, -1))
        assert flat_gt_pos_prob.sum() == b, "Each ground truth distribution needs to be a simplex"
        flat_pos_pred_logdistr = pos_pred_log_distr.view((b, -1))
        pos_loss = -(flat_gt_pos_prob * flat_pos_pred_logdistr).sum(dim=1).mean(dim=0)

        # Loss for predicting the yaw at the goal position
        gt_yaw_prob = gt_pos_yaw_prob
        yaw_loss = -(gt_yaw_prob * yaw_log_distr).sum(dim=(1, 2, 3)).mean(dim=0) # Sum across spatial dims, because only one spatial position is actually non-zero per batch

        # Loss for predicting the pitch at the goal pose (position + yaw)
        gt_pitch = gt_pitch.sum(dim=1, keepdim=True)
        has_pitch = (gt_pitch != 0)
        pitch_loss = (has_pitch * ((pitch_prediction - gt_pitch) ** 2)).sum() / (has_pitch.sum())

        loss = pos_loss + yaw_loss + pitch_loss

        viz = True

        def map_colors_for_viz(pdist):
                                   # Red - 0,  Blue - 1,    bluegreen - 2,    Yellow - 3
            colors = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 0.5, 0.5], [0.5, 0.5, 0]], dtype=pdist.dtype, device=pdist.device)
            # B x 4 x 3 x H x W
            colors = colors[None, :, :, None, None]
            pdist = pdist[:, :, None, :, :]
            pdist_c = (pdist * colors).sum(dim=1).clamp(0, 1)
            return pdist_c

        if viz and self.iter.item() % 1 == 0:
            alpha = 0.15

            # Position
            td_state_np = state_images[0].permute((1, 2, 0)).detach().cpu().numpy()
            td_g_np = gt_pos_prob[0].permute((1, 2, 0)).detach().cpu().numpy()
            comb_img_gt = alpha * td_state_np + (1 - alpha) * td_g_np
            show_image(comb_img_gt, "state_with_pos_gt", scale=4, waitkey=1)

            location_distr = torch.exp(pos_pred_log_distr)
            location_distr = location_distr / torch.max(location_distr)
            location_distr_np = location_distr[0].permute((1, 2, 0)).detach().cpu().numpy()
            comb_img_pred = td_state_np * alpha + (1 - alpha) * location_distr_np
            show_image(comb_img_pred, "state_with_pos_pred", scale=4, waitkey=1)

            # Yaw
            td_yaw_gt_np = map_colors_for_viz(gt_pos_yaw_prob)[0].permute((1, 2, 0)).detach().cpu().numpy()
            comb_yaw_img_gt = alpha * td_state_np + (1 - alpha) * td_yaw_gt_np
            show_image(comb_yaw_img_gt, "state_with_yaw_gt", scale=4, waitkey=1)

            yaw_distr_pred = torch.exp(yaw_log_distr)
            yaw_distr_pred = yaw_distr_pred / torch.max(yaw_distr_pred)
            yaw_distr_pred_np = map_colors_for_viz(yaw_distr_pred)[0].permute((1, 2, 0)).detach().cpu().numpy()
            comb_yaw_img_pred = td_state_np * alpha + (1 - alpha) * yaw_distr_pred_np
            show_image(comb_yaw_img_pred, "state_with_yaw_pred", scale=4, waitkey=1)

            # Pitch
            PITCH4 = False
            if PITCH4:
                # TODO: this should look up the pitch for the corresponding predicted / gt yaw bins. Too much work...
                td_state_np_tiled = np.tile(td_state_np, (4, 1, 1))
                gt_pitch_img = torch.cat([gt_pitch[:, 0:1], gt_pitch[:, 1:2], gt_pitch[:, 2:3], gt_pitch[:, 3:4]], dim=2)
                #gt_pitch_img = gt_pitch.sum(dim=1, keepdim=True)
                gt_pitch_img = torch.cat([gt_pitch_img.clamp(0, math.pi) * 0.5, gt_pitch_img.clamp(-math.pi, 0) * (-0.5), gt_pitch_img.clamp(0, 0)], dim=1)
                gt_pitch_img = gt_pitch_img[0].permute((1, 2, 0)).detach().cpu().numpy()
                comb_pitch_gt = alpha * td_state_np_tiled + (1 - alpha) * gt_pitch_img
                show_image(comb_pitch_gt, "state_with_pitch_gt", scale=4, waitkey=1)

                pitch_pred_img = torch.cat([pitch_prediction[:, 0:1], pitch_prediction[:, 1:2], pitch_prediction[:, 2:3], pitch_prediction[:, 3:4]], dim=2)
                pitch_pred_img = torch.cat([pitch_pred_img.clamp(0, 2) * 0.5, pitch_pred_img.clamp(-2, 0) * (-0.5), pitch_pred_img.clamp(0, 0)], dim=1)
                pitch_pred_img = pitch_pred_img[0].permute((1, 2, 0)).detach().cpu().numpy()
                comb_pitch_pred = alpha * td_state_np_tiled + (1 - alpha) * pitch_pred_img
                show_image(comb_pitch_pred, "state_with_pitch_pred", scale=4, waitkey=1)
            else:
                # TODO: this should look up the pitch for the corresponding predicted / gt yaw bins. Too much work...
                gt_pitch_img = gt_pitch
                #gt_pitch_img = gt_pitch.sum(dim=1, keepdim=True)
                gt_pitch_img = torch.cat([gt_pitch_img.clamp(0, math.pi) * 0.5, gt_pitch_img.clamp(-math.pi, 0) * (-0.5), gt_pitch_img.clamp(0, 0)], dim=1)
                gt_pitch_img = gt_pitch_img[0].permute((1, 2, 0)).detach().cpu().numpy()
                comb_pitch_gt = alpha * td_state_np + (1 - alpha) * gt_pitch_img
                show_image(comb_pitch_gt, "state_with_pitch_gt", scale=4, waitkey=1)

                pitch_pred_img = pitch_prediction
                pitch_pred_img = torch.cat([pitch_pred_img.clamp(0, math.pi) * 0.5, pitch_pred_img.clamp(-math.pi, 0) * (-0.5), pitch_pred_img.clamp(0, 0)], dim=1)
                pitch_pred_img = pitch_pred_img[0].permute((1, 2, 0)).detach().cpu().numpy()
                comb_pitch_pred = alpha * td_state_np + (1 - alpha) * pitch_pred_img
                show_image(comb_pitch_pred, "state_with_pitch_pred", scale=4, waitkey=1)

        #metrics = self.collect_metrics(location_log_distr, gt_location_flat)
        metrics = {}
        metrics["loss"] = loss.item()
        metrics["pitch_loss"] = pitch_loss.item()
        metrics["pos_loss"] = pos_loss.item()
        metrics["yaw_loss"] = yaw_loss.item()

        self.iter += 1

        return loss, metrics


import lgp.model_registry
lgp.model_registry.register_model("alfred_spatial_navigation_model", HlsmNavigationModel)
