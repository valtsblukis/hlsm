from typing import Iterable, Dict, List

import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer

from lgp.abcd.repr.task_repr import TaskRepr
from lgp.abcd.functions.task_repr_function import TaskReprFunction

from lgp.env.alfred.tasks import AlfredTask


MAX_STRLEN = 50

tokenizer = None


class HlsmTaskReprFunction(TaskReprFunction):
    def __init__(self):
        super().__init__()

    def _make_tokenizer(self):
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def _get_tokenizer(self) -> "PreTrainedTokenizer":
        global tokenizer
        if tokenizer is None:
            tokenizer = self._make_tokenizer()
        return tokenizer

    def forward(self, tasks: List[AlfredTask], device="cpu") -> "HlsmTaskRepr":
        tokenizer = self._get_tokenizer()
        str_repr = [str(t) for t in tasks]
        if callable(tokenizer):
            tokens = tokenizer(str_repr, padding='max_length', truncation=True, max_length=MAX_STRLEN, return_tensors="pt")
        else:
            tokens = tokenizer.encode(str_repr, padding='max_length', truncation=True, max_length=MAX_STRLEN, return_tensors="pt")
        tokens = tokens.to(device)
        return HlsmTaskRepr(tokens, text=str_repr)

    def inverse_to_str(self, task_repr: "HlsmTaskRepr") -> str:
        tokenizer = self._get_tokenizer()
        task_tokens = tokenizer.convert_ids_to_tokens(task_repr.data)
        task_str = tokenizer.convert_tokens_to_string(task_tokens)
        # TODO: Figure how to disambiguate a batch from a single string
        return task_str


class HlsmTaskRepr(TaskRepr):
    """
    Represents a task specified in natural language as a tensor of integer tokens
    """

    def __init__(self, data: torch.tensor, text=None):
        super().__init__()
        self.data = data
        self.text = text

    def to(self, device=None):
        self.data = self.data.to(device)
        return self

    def as_tensor(self):
        return self.data

    def __getitem__(self, item):
        return HlsmTaskRepr(self.data[item])

    @classmethod
    def from_task(cls, task: AlfredTask, device="cpu") -> "HlsmTaskRepr":
        return HlsmTaskReprFunction()(task, device)

    def __str__(self):
        return HlsmTaskReprFunction().inverse_to_str(self)

    @classmethod
    def collate(cls, states: Iterable["HlsmTaskRepr"]) -> "HlsmTaskRepr":
        """
        Creates a single Action that represents a batch of actions
        """
        datas = torch.cat([s.data for s in states], dim=0)
        return cls(datas)

    def represent_as_image(self) -> torch.tensor:
        image = utils.one_hot_to_image(self.data)
        return image