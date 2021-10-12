from typing import Dict, List, Union

import torch
import torch.nn as nn

from lgp.abcd.functions.action_proposal import ActionProposal
from lgp.abcd.model import LearnableModel

import lgp.env.alfred.segmentation_definitions as segdef
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal

from lgp.ops.spatial_distr import multidim_logsoftmax

from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from lgp.models.alfred.hlsm.hlsm_task_repr import HlsmTaskRepr

from lgp.models.alfred.hlsm.transformer_modules.subgoal_history_encoder import SubgoalHistoryEncoder
from lgp.models.alfred.hlsm.transformer_modules.state_repr_encoder_pooled import StateReprEncoderPooled
from lgp.models.alfred.hlsm.transformer_modules.language_encoder import BERTLanguageEncoder
from lgp.models.alfred.hlsm.transformer_modules.action_predictor import ActionPredictor

from lgp.models.alfred.hlsm.unets.lingunet_3 import Lingunet3
from lgp.models.alfred.voxel_grid import VoxelGrid

from lgp.ops.misc import batched_index_select

from lgp.utils.viz import show_image
from lgp.flags import GLOBAL_VIZ

from lgp.parameters import Hyperparams


class HlsmSubgoalModel(ActionProposal, LearnableModel):
    class ModelState(ActionProposal.ModelState):
        def __init__(self):
            self.action_history = []
            self.logged_failure = False
            self.step = 0

        def action_execution_failed(self):
            # Remove the last action from the action history and allow a re-try.
            if not self.logged_failure:
                print("                                            LOGGING SKILL FAILURE")
                self.action_history = self.action_history[:-1]
                self.logged_failure = True

        def log_action(self, action):
            # When we log a new predicted action, reset the logged_failure flag so that if this action fails,
            # it can be removed from the action history
            self.logged_failure = False
            self.action_history.append(action)
            print("                                             LOGGING NEW ACTION")

        @classmethod
        def blank(cls):
            return None

    def __init__(self, hparams: Hyperparams):
        super().__init__()

        self.action_type_dim = AlfredSubgoal.get_action_type_space_dim()
        self.data_c = AlfredSpatialStateRepr.get_num_data_channels()
        self.hidden_dim = 128

        self.no_posemb_baseline = hparams.get("no_posemb_baseline", False)
        self.no_acthist_baseline = hparams.get("no_acthist_baseline", False)
        self.no_vision_baseline = hparams.get("no_vision_baseline", False)
        self.no_language_baseline = hparams.get("no_language_baseline", False)

        print("SpatialTransformerModel2 baseline config:"
              f"No vision: {self.no_vision_baseline}"
              f"No language: {self.no_language_baseline}"
              f"No posemb: {self.no_posemb_baseline}"
              f"No acthist: {self.no_acthist_baseline}")

        # Networks / Models
        if not self.no_language_baseline:
            self.language_encoder = BERTLanguageEncoder(self.hidden_dim)

        if not self.no_vision_baseline:
            self.state_repr_encoder = StateReprEncoderPooled(self.hidden_dim)

        self.action_history_encoder = SubgoalHistoryEncoder(self.hidden_dim,
                                                            ablate_no_acthist=self.no_acthist_baseline,
                                                            ablate_no_posemb=self.no_posemb_baseline)

        self.mask_model = Lingunet3(2 * self.action_type_dim + AlfredSpatialStateRepr.get_2d_feature_dim(),
                                    self.hidden_dim, 1)

        self.action_predictor = ActionPredictor(self.hidden_dim, joint_prob=True)

        self.nllloss = nn.NLLLoss(reduce=True, size_average=True)

        self.act = nn.LeakyReLU()
        self.iter_step = 0
        self.model_state = None
        self.log_internal_activations = True
        self.trace = {
            "subgoal": None
        }
        self.metrics = {}

        self.reset_state()

    def set_log_internal_activations(self, enable):
        self.log_internal_activations = enable

    def _get_state_for_action(self, action: AlfredSubgoal) -> "ActionProposal.ModelState":
        # TODO: If we want to do any action-conditioned reasoning, do it here.
        # TODO: Make sure to not increment model_state.step in two different places (here and forward)
        return self.model_state

    def get_trace(self, device="cpu") -> Dict:
        return {k: v.to(device) if v is not None else v for k, v in self.trace.items()}

    def clear_trace(self):
        ...
        #self.trace = {}

    def action_execution_failed(self):
        self.model_state.action_execution_failed()

    def log_action(self, action: AlfredSubgoal):
        self.model_state.log_action(action)

    def get_state(self) -> "HlsmSubgoalModel.ModelState":
        return self.model_state

    def set_state(self, state: "HlsmSubgoalModel.ModelState"):
        self.model_state = state

    def reset_state(self):
        self.model_state = HlsmSubgoalModel.ModelState()
        self.trace = {
            "subgoal": None
        }

    def _argmax_action(self, type_distr, arg_vectors):
        act_type_id = torch.argmax(type_distr, dim=1)
        act_type_id = act_type_id[0].item()
        act_type_str = AlfredSubgoal.action_type_intid_to_str(act_type_id)
        # TODO: Check for HL->Subgoal
        arg_vector = arg_vectors[:, act_type_id, :]

        # Computed for debugging purposes only
        top5_objects = [(segdef.object_intid_to_string(x.item() - 1), arg_vector[0, x.item()].item()) for x in reversed(arg_vector[0].argsort()[-5:])]
        # print(f"Top5 objects: {top5_objects}")

        pass_objects = arg_vector > 0.04
        arg_vector = arg_vector * pass_objects
        arg_vector /= (arg_vector.sum() + 1e-10)
        return AlfredSubgoal.from_type_str_and_arg_vector(act_type_str, arg_vector)

    def _sample_subgoal(self, type_distr, arg_vectors):
        act_type_id = torch.distributions.Categorical(type_distr).sample().item()
        act_type_str = AlfredSubgoal.action_type_intid_to_str(act_type_id)
        arg_vector = arg_vectors[:, act_type_id, :]

        top5_types = [(AlfredSubgoal.action_type_intid_to_str(a.item()), type_distr[0, a.item()].item()) for a in reversed(type_distr[0].argsort()[-5:])]
        print(f"Top5 types: {top5_types}")

        # Computed for debugging purposes only
        top5_objects = [(segdef.object_intid_to_string(x.item() - 1), arg_vector[0, x.item()].item()) for x in reversed(arg_vector[0].argsort()[-5:])]
        print(f"Top5 objects: {top5_objects}")
        print(f"Action history: {[str(a) for a in self.model_state.action_history]}")

        # Zero out the long tail - otherwise that contains most of the prob mass which doesn't make sense.
        pass_objects = arg_vector > 0.04
        arg_vector = arg_vector * pass_objects
        arg_vector /= (arg_vector.sum() + 1e-10)

        act_arg_id = torch.distributions.Categorical(arg_vector).sample().item()
        arg_vector_out = torch.zeros_like(arg_vector)
        arg_vector_out[0, act_arg_id] = 1.0
        return AlfredSubgoal.from_type_str_and_arg_vector(act_type_str, arg_vector_out)

    def mle(self,
            state: AlfredSpatialStateRepr,
            task: HlsmTaskRepr,
            model_state: "HlsmSubgoalModel.ModelState"):
        return self.forward_inference(state, task, model_state)

    # ---------------------------------------------------------------------------------

    def _log_activations(self, states, act_type_distr, act_arg_distr, action):
        if self.log_internal_activations:
            with torch.no_grad():
                act_arg_distr_argmax_type = act_arg_distr[:, act_type_distr[0].argmax(), :]
                state_kernel = act_arg_distr_argmax_type[:, 1:].clone()
                state_data = states.data.data  # B x C x W x L x H

                state_response = torch.einsum("bcwlh,bc->bwlh", state_data.float(), state_kernel.float())  # B x W x L x H
                action_arg_distribution_log = multidim_logsoftmax(state_response, dims=(1, 2, 3))
                # Replicate - use the same argument distribution for "all action types"
                action_arg_distribution_log = action_arg_distribution_log[:, None, :, :, :].repeat(
                    (1, self.action_type_dim, 1, 1, 1))

                #self.trace["state_repr"] = states
                #self.trace["filters"] = state_kernel
                #self.trace["action_type_distribution"] = torch.exp(act_type_distr)
                #self.trace["action_arg_distribution"] = torch.exp(action_arg_distribution_log)
                self.trace["subgoal"] = action
        return

    def forward_inference(self,
                          states: AlfredSpatialStateRepr,
                          tasks: HlsmTaskRepr,
                          model_state: "HlsmSubgoalModel.ModelState"):
        device = states.data.data.device

        action_history = model_state.action_history

        # The most recent timestep doesn't have an action label - that's what we're predicting.
        # Add a dummy tensor there. The downstream model doesn't look at it anyway (it's masked out in attention masks)
        current_action_label = torch.zeros((1, 2), device=device, dtype=torch.int64)
        current_action_label[0, 0] = 2
        current_action_label[0, 1] = 33

        action_labels = torch.cat([a.to_tensor(device=device, dtype=torch.int64) for a in action_history] + [current_action_label], dim=0)
        batch_id = [0 for _ in range(len(action_labels))]

        act_type_logprobs, act_arg_logprobs, task_emb = self._forward_model(states, tasks, action_labels, batch_id)

        # We care about predicting the CURRENT action (the last one),
        # even though we are running the model on the entire action history sequence.
        act_type_logprob = act_type_logprobs[-1:, :]
        act_arg_logprob = act_arg_logprobs[-1:, :]

        type_distr = torch.exp(act_type_logprob)
        arg_distr = torch.exp(act_arg_logprob)
        arg_distr = arg_distr / arg_distr.sum(dim=2, keepdim=True)  # Re-normalize
        subgoal = self._sample_subgoal(type_distr, arg_distr)

        arg_mask_3d = self.forward_mask(states, task_emb, subgoal, self.model_state.action_history, batch_training=False)

        arg_mask_voxelgrid = VoxelGrid(arg_mask_3d, arg_mask_3d, states.data.voxel_size, states.data.origin)
        subgoal.argument_mask = arg_mask_voxelgrid

        # Create some graphics for the gifs
        self._log_activations(states, torch.exp(act_type_logprob), torch.exp(act_arg_logprob), subgoal)

        return subgoal

    def _forward_model(self,
                       states: Union[AlfredSpatialStateRepr, torch.tensor],
                       tasks: HlsmTaskRepr,
                       sem_actions: torch.tensor,
                       batch_id: List[int]):
        # sem_actions: B x 2
        bs = states.data.data.shape[0]
        device = states.data.data.device

        # TxD_{u}
        if self.no_language_baseline:
            task_embeddings = torch.zeros((bs, self.hidden_dim), device=device, dtype=torch.float32)
        else:
            task_embeddings = self.language_encoder(tasks)

        # TxD_{a}
        action_hist_embeddings = self.action_history_encoder(sem_actions, batch_id)

        # TxD_{s}
        if self.no_vision_baseline:
            state_embeddings = torch.zeros((bs, self.hidden_dim), device=device)
        else:
            state_embeddings = self.state_repr_encoder(states, task_embeddings, action_hist_embeddings)

        # Drop the last action history embedding
        action_hist_embeddings = action_hist_embeddings[:-1]

        # If we're running inference, we want to predict the most recent action from the most recent state.
        # Take the most recent action embedding
        ns = state_embeddings.shape[0]
        action_hist_embeddings = action_hist_embeddings[-ns:]

        act_type_logprob, act_arg_logprob = self.action_predictor(
            state_embeddings, task_embeddings, action_hist_embeddings)

        return act_type_logprob, act_arg_logprob, task_embeddings

    def forward_mask(self,
                     state_repr,
                     task_emb : torch.tensor,
                     action: AlfredSubgoal,
                     action_history: List[AlfredSubgoal],
                     batch_training=False):

        # STATE REPRESENTATION
        state_features = state_repr.get_nav_features_2d()

        # ACTION HISTORY REPRESENTATION
        if len(action_history) > 0:
            action_history = AlfredSubgoal.collate(action_history)

            past_action_types = action_history.type_oh()
            past_action_masks = action_history.get_argument_mask()
            past_action_typed_mask_3d = past_action_types[:, :, None, None, None] * past_action_masks
            action_history_typed_masks_2d = past_action_typed_mask_3d.cumsum(dim=0).max(dim=4).values
        else:
            b, f, h, w = state_features.shape
            ac = AlfredSubgoal.get_action_type_space_dim()
            action_history_typed_masks_2d = torch.zeros((b, ac, h, w), device=state_features.device)

        # PROPOSED ACTION REPRESENTATION
        # Build proposal of current action type and arg
        proposed_action_masks = action.build_spatial_arg_proposal(state_repr)
        proposed_action_masks_2d = proposed_action_masks.max(dim=4).values
        proposed_action_types = action.type_oh()
        proposed_typed_masks_2d = proposed_action_types[:, :, None, None] * proposed_action_masks_2d

        if batch_training:
            # Roll action histories forward so that the model can't peek at curent action argument masks
            action_history_typed_masks_2d = action_history_typed_masks_2d.roll(shifts=1, dims=0).clone()
            # The last action rolls back to the start - zero it out.
            action_history_typed_masks_2d[0] = 0
        else:
            # Take the last action history mask representation
            action_history_typed_masks_2d = action_history_typed_masks_2d[-1:]

        # Run the mask prediction LingUNet
        x = torch.cat([action_history_typed_masks_2d, proposed_typed_masks_2d, state_features], dim=1)
        ctx = task_emb
        pred_masks_2d = self.mask_model(x, ctx)
        pred_logprobs_2d = multidim_logsoftmax(pred_masks_2d, dims=(2, 3))

        if batch_training:
            # During training, learn to predict masks in top-down view. No need to go to 3D
            return pred_logprobs_2d, proposed_action_masks_2d
        else:
            # At test-time, lift the mask to 3D by masking the proposals
            pred_probs_3d = proposed_action_masks * torch.exp(pred_logprobs_2d)[:, :, :, :, None]
            # Standardize against the peak activation
            pred_probs_3d = pred_probs_3d / (pred_probs_3d.max() + 1e-10)

            VIZ = GLOBAL_VIZ
            if VIZ:
                show_image(pred_probs_3d.max(dim=4).values[0].detach().cpu(), "Refined mask", waitkey=1, scale=4)
                show_image(proposed_action_masks_2d[0].detach().cpu(), "Proposed mask", waitkey=1, scale=4)

            return pred_probs_3d

    def get_name(self) -> str:
        return "alfred_spatial_transformer_model_2"

    def success(self, pred_logits, class_indices):
        amax_idx = pred_logits.argmax(1)
        target_idx = class_indices
        #print(amax_idx[0], target_idx[0])# == target_idx.sum())
        succ = (amax_idx == target_idx)
        return succ

    def collect_metrics(self, act_type_logprob, act_type_gt, act_arg_logprob, act_arg_gt, sem_actions, batch_id):
        # Metrics:
        type_step_success = self.success(act_type_logprob, act_type_gt)
        arg_step_success = self.success(act_arg_logprob, act_arg_gt)

        tensor_batchid = torch.tensor(batch_id, device=sem_actions.device)
        type_per_step_success_rate = type_step_success.sum().item() / type_step_success.shape[0]
        arg_per_step_success_rate = arg_step_success.sum().item() / arg_step_success.shape[0]
        act_per_step_success_rate = (type_step_success * arg_step_success).sum().item() / type_step_success.shape[0]

        type_full_correct = 0
        arg_full_correct = 0
        act_full_correct = 0
        num_b = max(batch_id) + 1

        for b in range(num_b):
            isb = (tensor_batchid == b)
            b_type_succ_cnt = (type_step_success * isb).sum()
            b_arg_succ_cnt = (arg_step_success * isb).sum()
            b_cnt = isb.sum()
            b_tc = (b_type_succ_cnt == b_cnt).item()
            b_ac = (b_arg_succ_cnt == b_cnt).item()
            type_full_correct += 1 if b_tc else 0
            arg_full_correct += 1 if b_ac else 0
            act_full_correct += 1 if b_ac * b_tc else 0

        type_sequence_success_rate = type_full_correct / num_b
        arg_sequence_success_rate = arg_full_correct / num_b
        act_sequence_success_rate = act_full_correct / num_b

        metrics = {
            "act_type_step_sr": type_per_step_success_rate,
            "act_arg_step_sr": arg_per_step_success_rate,
            "act_step_sr": act_per_step_success_rate,

            "act_type_seq_sr": type_sequence_success_rate,
            "act_arg_seq_sr": arg_sequence_success_rate,
            "act_seq_sr": act_sequence_success_rate
        }
        return metrics

    def loss(self, batch: Dict):
        # This is now forward
        return self.forward(batch)

    def forward(self, batch: Dict):
        if batch["states"] is None:
            states = batch["states_preproc"]
        else:
            states = batch["states"]
        tasks = batch["task_reprs"]
        subgoals_gt : AlfredSubgoal = batch["subgoals"]
        batch_id = batch["batch_id"]
        actions_gt_sem_tensor = subgoals_gt.to_tensor()

        act_type_logprob, act_arg_logprob, task_emb = self._forward_model(states, tasks, actions_gt_sem_tensor, batch_id)

        act_type_gt = actions_gt_sem_tensor[:, 0]
        act_arg_gt = actions_gt_sem_tensor[:, 1] + 1

        # For each batch element, grab the argument distribution corresponding to the ground truth action type
        act_arg_logprob = batched_index_select(act_arg_logprob, dim=1, index=act_type_gt)[:, 0, :]

        # Predict action argument masks
        action_history_gt: List[AlfredSubgoal] = subgoals_gt.disperse()
        act_mask_pred_logprob_2d, act_mask_proposed_2d = self.forward_mask(
            states, task_emb, subgoals_gt, action_history_gt, batch_training=True)

        # It only makes sense to learn action argument prediction over observed space
        obs_mask = states.get_observability_map_2d()

        act_mask_pred_prob_2d = torch.exp(act_mask_pred_logprob_2d)
        act_mask_gt_2d = subgoals_gt.get_argument_mask().max(dim=4).values
        act_mask_gt_2d = act_mask_gt_2d * obs_mask
        domain_size = act_mask_gt_2d.sum(dim=(1, 2, 3), keepdims=True)
        act_mask_gt_2d = act_mask_gt_2d / (domain_size + 1e-10)
        has_arg_mask = act_mask_gt_2d.sum(dim=(1, 2, 3)) > 0

        # TODO: Sem actions should be AlfredActionHLSem
        type_loss = self.nllloss(input=act_type_logprob, target=act_type_gt)
        arg_loss = self.nllloss(input=act_arg_logprob, target=act_arg_gt)

        # Spatial cross-entropy loss:
        argmask_loss = -((act_mask_gt_2d * act_mask_pred_logprob_2d).sum(dim=(2, 3)) * has_arg_mask).mean()
        # BCE loss:
        #argmask_loss = -((act_mask_gt_2d * torch.log(act_mask_pred_2d)).sum(dim=(1, 2, 3)) / (
        #    act_mask_gt_2d.sum(dim=(1, 2, 3)) + 1e-10)).mean()
        loss = type_loss + arg_loss + argmask_loss

        metrics = self.collect_metrics(act_type_logprob, act_type_gt, act_arg_logprob, act_arg_gt, actions_gt_sem_tensor, batch_id)

        metrics["loss"] = loss.detach().cpu().item()
        metrics["type_loss"] = type_loss.detach().cpu().item()
        metrics["arg_loss"] = arg_loss.detach().cpu().item()
        metrics["argmask_loss"] = argmask_loss.detach().cpu().item()

        VIZ = GLOBAL_VIZ
        if VIZ:
            with torch.no_grad():
                mask_viz = torch.cat([act_mask_gt_2d[0], act_mask_proposed_2d[0], act_mask_pred_prob_2d[0] * domain_size[0]], dim=0).clamp(0, 1)
                mask_viz = mask_viz * has_arg_mask[0] # Just blank out examples where there are no argument labels
                mask_viz_np = mask_viz.permute((1, 2, 0)).detach().cpu().numpy()
                show_image(mask_viz_np, "R: gt, G: proposal, B: refined pred", scale=4, waitkey=1)

                state_viz = states.get_nav_features_2d_viz()[0].permute((1, 2, 0)).detach().cpu().numpy()
                show_image(state_viz, "State features", scale=4, waitkey=1)

        return loss, metrics


import lgp.model_registry
lgp.model_registry.register_model("alfred_subgoal_model", HlsmSubgoalModel)
