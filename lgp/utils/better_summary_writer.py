from tensorboardX import SummaryWriter


class BetterSummaryWriter(SummaryWriter):

    def __init__(self, *args, start_iter=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = start_iter

    def inc_iter(self):
        self.iter += 1

    def add_scalar_dict(self, tag, d, iter_num=None):
        if not iter_num:
            iter_num = self.iter
        for k, v in d.items():
            subtag = f"{tag}/{k}"
            self.add_scalar(subtag, v, iter_num)