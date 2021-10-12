import torch
import math


def padded_roll_2d(inp, sy, sx):
    # Given a 4-dimensional input of shape BxCxHxW, roll it along two dimensions H,W by (sy,sx), but do not roll values over edges.
    b, c, h, w = inp.shape

    # Can do with a smaller canvas
    if 0 <= math.fabs(sy) < h // 2 and 0 <= math.fabs(sx) < w // 2:
        canvas = torch.zeros((b, c, h*2, w*2), dtype=inp.dtype, device=inp.device)
        canvas[:, :, h//2:3*h//2, w//2:3*w//2] = inp
        canvas = torch.roll(canvas, shifts=(sy, sx), dims=(2, 3))
        outp = canvas[:, :, h//2:3*h//2, w//2:3*w//2]
    # Need a bigger canvas
    else:
        canvas = torch.zeros((b, c, h * 4, w * 4), dtype=inp.dtype, device=inp.device)
        canvas[:, :, 3*h//2:5*h//2, 3*w//2:5*w//2] = inp
        canvas = torch.roll(canvas, shifts=(sy, sx), dims=(2, 3))
        outp = canvas[:, :, 3*h//2:5*h//2, 3*w//2:5*w//2]

    return outp


def batch_id_to_range(batch_id, device, dtype):
    b = len(batch_id)
    rng = torch.zeros([b], device=device, dtype=dtype)
    ord = 0
    prev_bid = -1
    for i, b_id in enumerate(batch_id):
        if b_id != prev_bid:
            ord = 0
        rng[i] = ord
        ord += 1
        prev_bid = b_id
    return rng


def index_to_onehot(index_tensor, num_classes):
    if len(index_tensor.shape) == 1:
        index_tensor = index_tensor[:, None]
    bs = index_tensor.shape[0]
    device = index_tensor.device

    ones = torch.ones((bs,), device=device)
    oh = torch.zeros((bs, num_classes), device=device)
    oh = oh.scatter_add(dim=1, index=index_tensor, src=ones[:, None])
    return oh


def onehot_to_index():
    # TODO
    raise NotImplementedError()


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)