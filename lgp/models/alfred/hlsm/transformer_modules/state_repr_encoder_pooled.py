from typing import Union

import torch
import torch.nn as nn

from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
#from lgp.models.alfred.hlsm.transformer_modules.transformer_layer import TransformerSideLayer


class StateReprEncoderPooled(nn.Module):

    def __init__(self, dmodel, nhead=8):
        super().__init__()
        self.task_layer = nn.Linear(dmodel, dmodel * nhead)
        self.state_layer = nn.Linear(AlfredSpatialStateRepr.get_num_data_channels() * 2, dmodel)
        #self.enc_layer_3d = TransformerSideLayer(d_model=dmodel, n_head=1, dim_ff=dmodel, kvdim=dmodel)
        #self.enc_layer_1d = TransformerSideLayer(d_model=dmodel, n_head=1, dim_ff=dmodel, kvdim=dmodel)

        self.dmodel = dmodel
        self.nhead = nhead

    @classmethod
    def _make_pooled_repr(cls, state_reprs):
        b, c, w, l, h = state_reprs.data.data.shape
        state_pooled = state_reprs.data.data.view([b, c, w*l*h]).max(dim=2).values
        state_pooled_and_inv = torch.cat([state_pooled, state_reprs.inventory_vector], dim=1)
        return state_pooled_and_inv

    def forward(self,
                state_reprs: Union[AlfredSpatialStateRepr, torch.tensor],
                task_embeddings: torch.tensor,
                action_hist_embeddings: torch.tensor
                ) -> torch.tensor:
        """
        Args:
            state_reprs: AlfredSpatialStateRepr with data of shape BxCxWxLxH
            task_repr: AlfredSpatialTaskRepr with data of shape BxD_{u}
            action_hist_repr: torch.tensor of shape BxD_{a}

        Returns:
            BxD_{s} dimensional batch of state representation embeddings.
        """
        if isinstance(state_reprs, AlfredSpatialStateRepr):
            state_pooled_and_inv = StateReprEncoderPooled._make_pooled_repr(state_reprs)
        else:
            state_pooled_and_inv = state_reprs
        flat_repr = self.state_layer(state_pooled_and_inv.float())
        return flat_repr
