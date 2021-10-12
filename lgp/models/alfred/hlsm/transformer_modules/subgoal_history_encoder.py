import torch
import torch.nn as nn

from lgp.models.alfred.hlsm.transformer_modules.positional_encodings import positional_encoding_1d_flat
from lgp.models.alfred.hlsm.transformer_modules.transformer_layer import TransformerEncoderLayer

from lgp.env.alfred.alfred_subgoal import AlfredSubgoal

from lgp.env.alfred import segmentation_definitions as segdef

from lgp.ops.misc import index_to_onehot


class SubgoalHistoryEncoder(nn.Module):

    def __init__(self, dmodel, ablate_no_acthist=False, ablate_no_posemb=False):
        super().__init__()
        self.num_actions = AlfredSubgoal.get_action_type_space_dim()
        self.num_objects = segdef.get_num_objects()
        self.dim = dmodel

        self.ablate_no_acthist = ablate_no_acthist
        self.ablate_no_posemb = ablate_no_posemb

        self.type_linear = nn.Linear(self.num_actions, self.dim)
        self.arg_linear = nn.Linear(self.num_objects + 1, self.dim)
        self.sos_token_emb = nn.Parameter(torch.zeros(self.dim), requires_grad=True)

        self.transformer_layer_a = TransformerEncoderLayer(
            d_model=self.dim,
            n_head=8,
            dim_ff=self.dim
        )
        self.transformer_layer_b = TransformerEncoderLayer(
            d_model=self.dim,
            n_head=8,
            dim_ff=self.dim
        )

    def _build_attn_mask(self, batch_id, b, incl_self=False):
        is_b = (batch_id == b)
        n = torch.sum(is_b)
        mask = torch.ones((n, n), device=batch_id.device, dtype=torch.float32)

        # Allow attending to input position
        if incl_self:
            mask = torch.triu(mask, diagonal=0)
        # Only allow attending to strictly previous positions.
        else:
            mask = torch.triu(mask, diagonal=1)
        return mask

    def _build_attn_masks(self, batch_id, device, add_sos_tok=False, incl_self=False):
        bt = len(batch_id)
        num_batches = max(batch_id) + 1 if len(batch_id) > 0 else 0
        batch_id = torch.tensor(batch_id, device=device)

        # Build attention masks
        attn_mask = torch.zeros((bt, bt), device=device, dtype=torch.float32)
        s = 0
        for b in range(num_batches):
            mask = self._build_attn_mask(batch_id, b, incl_self)
            n = mask.shape[0]
            attn_mask[s:s+n, s:s+n] = mask
            s = s+n
        # Rows correspond to elements in output sequence. Columns to elements in input sequence
        attn_mask = attn_mask.T

        # Every output element may attend to the first element in the input sequence
        # The first element represents the "SOS" token
        if add_sos_tok:
            attn_mask = torch.cat([torch.zeros([1, bt], device=device, dtype=torch.float32), attn_mask], dim=0)
            attn_mask = torch.cat([torch.ones([bt+1, 1], device=device, dtype=torch.float32), attn_mask], dim=1)

        # This is undocumented, but internally PyTorch MultiheadAttention uses masked_fill to zero-out attention
        # weights corresponding to the masked elements.
        # That means that True values are masked-out, and False values are left untouched. Grrrr.
        attn_mask = 1 - attn_mask
        return attn_mask.bool()

    def forward(self, action_seq, batch_id):
        """
        Args:
            action_seq: Tx2-dimensional long tensor
            batch_id: T-dimensional long-tensor indicating which batch element does each action belong to.

        Returns:
            TxD_{a}-dimensional tensor of action history embeddings at each timestep of action_seq
        """
        device = action_seq.device

        type_oh = index_to_onehot(action_seq[:, 0:1], self.num_actions)
        arg_oh = index_to_onehot(action_seq[:, 1:2] + 1, self.num_objects + 1)
        if self.ablate_no_acthist:
            type_oh = torch.zeros_like(type_oh)
            arg_oh = torch.zeros_like(arg_oh)

        type_emb = self.type_linear(type_oh)
        arg_emb = self.arg_linear(arg_oh)

        pos_enc = positional_encoding_1d_flat(type_emb, batch_id)
        if self.ablate_no_posemb:
            pos_enc = torch.zeros_like(pos_enc)

        act_emb = type_emb + arg_emb + pos_enc
        start_and_act_emb = torch.cat([self.sos_token_emb[None, :], act_emb], dim=0)

        # Allow attending only to strictly previous actions (not current action)
        self_attn_masks_a = self._build_attn_masks(batch_id, device, add_sos_tok=True, incl_self=True)
        # Allow attending to current representation that only depends on previous actions.
        self_attn_masks_b = self._build_attn_masks(batch_id, device, add_sos_tok=True, incl_self=True)

        # self_attn_masks indicates for each column, whether the corresponding
        # row maps to a previous action in the same rollout.
        # Rows and columns are both over the sequence of actions.
        #       sos, a_0, a_1, a_2, a_3, a_4
        # sos   1    0    0    0    0    0
        # a_0   1    0    0    0    0    0
        # a_1   1    1    0    0    0    0
        # a_2   1    1    1    0    0    0
        # a_3   1    0    0    0    0    0
        # a_4   1    0    0    0    1    0

        # Two transformer layers
        enc_seq_a, aw_a = self.transformer_layer_a(start_and_act_emb, attn_mask=self_attn_masks_a, return_attn=True)
        enc_seq_b, aw_b = self.transformer_layer_b(enc_seq_a, attn_mask=self_attn_masks_b, return_attn=True)
        return enc_seq_b
