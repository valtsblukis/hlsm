import torch
import torch.nn as nn
import torch.nn.functional as F

import lgp.env.alfred.segmentation_definitions as segdef
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal


class ActionPredictor(nn.Module):
    def __init__(self, dmodel, joint_prob=False):
        super().__init__()
        self.num_types = AlfredSubgoal.get_action_type_space_dim()
        self.num_args = segdef.get_num_objects() + 1
        self.joint_prob = joint_prob
        self.linear_a = nn.Linear(dmodel * 3, dmodel)
        self.linear_a1 = nn.Linear(dmodel, dmodel)
        self.linear_a2 = nn.Linear(dmodel * 2, dmodel)

        if self.joint_prob:
            self.linear_b = nn.Linear(dmodel * 3, self.num_types + self.num_args * self.num_types)
        else:
            self.linear_b = nn.Linear(dmodel * 3, self.num_types + self.num_args)
        self.act = nn.LeakyReLU()

    def forward(self, state_embeddings, sentence_embeddings, action_hist_embeddings):
        # If needed, broadcast the sentence embedding across time
        if sentence_embeddings.shape[0] == 1 and state_embeddings.shape[0] > 1:
            sentence_embeddings = sentence_embeddings.repeat((state_embeddings.shape[0], 1))

        combined_embedding = torch.cat([state_embeddings, sentence_embeddings, action_hist_embeddings], dim=1)

        x1 = self.act(self.linear_a(combined_embedding))
        x2 = self.act(self.linear_a1(x1))
        x12 = torch.cat([x1, x2], dim=1)
        x3 = self.act(self.linear_a2(x12))

        x123 = torch.cat([x1, x2, x3], dim=1)
        x = self.linear_b(x123)
        #x = self.linear_b(self.act(self.linear_a(combined_embedding)))

        act_type_logits = x[:, :self.num_types]
        act_arg_logits = x[:, self.num_types:]

        if self.joint_prob:
            # Output Type x Arg matrix of P(argument | type)
            b = act_arg_logits.shape[0]
            act_arg_logits = act_arg_logits.view([b, self.num_types, self.num_args])
            act_type_logprob = F.log_softmax(act_type_logits, dim=1)
            act_arg_logprob = F.log_softmax(act_arg_logits, dim=2)
        else:
            # Output P(argument), P(type) separately
            # TODO: This seems redundant given lines 41-42
            act_type_logits = x[:, :self.num_types]
            act_arg_logits = x[:, self.num_types:]

            act_type_logprob = F.log_softmax(act_type_logits, dim=1)
            act_arg_logprob = F.log_softmax(act_arg_logits, dim=1)

        return act_type_logprob, act_arg_logprob