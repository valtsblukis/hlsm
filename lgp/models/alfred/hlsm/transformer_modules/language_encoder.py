import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


from lgp.models.alfred.hlsm.hlsm_task_repr import HlsmTaskRepr


class BERTLanguageEncoder(nn.Module):

    def __init__(self, dmodel):
        super().__init__()
        self.bertmodel = AutoModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, dmodel)

    def forward(self, tasks: HlsmTaskRepr):
        bertmodeloutput = self.bertmodel(tasks.data.input_ids)
        last_hidden_states = bertmodeloutput.last_hidden_state # B x N x D

        pool_output = bertmodeloutput.pooler_output
        # This is a hack to make the pooler weights "used" as far as DistributedDataParallel is concered
        cls_hidden = last_hidden_states[:, 0, :] + 1e-20 * pool_output

        sentence_embedding = self.linear(cls_hidden)
        return sentence_embedding