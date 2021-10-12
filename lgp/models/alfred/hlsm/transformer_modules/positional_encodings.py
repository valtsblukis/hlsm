import torch
from lgp.ops.misc import batch_id_to_range


def positional_encoding_1d_flat(x: torch.tensor, batch_id: torch.tensor):
    b, c = x.shape
    assert c % 2 == 0
    inv_freq = 1.0 / (10000 ** (torch.arange(0, c, 2, device=x.device).float() / c))
    pos_b = batch_id_to_range(batch_id, x.device, x.dtype)
    sin_inp_h = torch.einsum("i,j->ij", pos_b, inv_freq)
    emb_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=-1)
    return emb_h


def positional_encoding_1d(x: torch.tensor):
    b, c, h = x.shape
    orig_c = c
    if c % 2:
        c += 1
    inv_freq = 1.0 / (10000 ** (torch.arange(0, c, 2, device=x.device).float() / c))
    dtype = inv_freq.dtype

    pos_h = torch.arange(h, device=x.device, dtype=dtype)
    sin_inp_h = torch.einsum("i,j->ij", pos_h, inv_freq)
    emb_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=-1)
    emb = torch.zeros((1, c, h), device=x.device, dtype=dtype)
    emb[0, :, :] = emb_h
    return emb[:, :orig_c, :].repeat((b, 1, 1))


def positional_encoding_2d(x: torch.tensor):
    b, c, w, h = x.shape
    orig_c = c
    if c % 2:
        c += 1
    inv_freq = 1.0 / (10000 ** (torch.arange(0, c, 2, device=x.device).float() / c))
    dtype = inv_freq.dtype

    pos_w = torch.arange(w, device=x.device, dtype=dtype)
    pos_h = torch.arange(h, device=x.device, dtype=dtype)
    sin_inp_w = torch.einsum("i,j->ij", pos_w, inv_freq)
    sin_inp_h = torch.einsum("i,j->ij", pos_h, inv_freq)
    emb_w = torch.cat((sin_inp_w.sin(), sin_inp_w.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
    emb_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=-1)
    emb = torch.zeros((1, c*2, w, h), device=x.device, dtype=dtype)
    emb[0, :c, :, :] = emb_w
    emb[0, c:, :, :] = emb_h
    return emb[:, :orig_c, :, :].repeat((b, 1, 1, 1))


def positional_encoding_3d(shape, device):
    b, c, w, l, h = shape
    orig_c = c
    if c % 2:
        c += 1
    inv_freq = 1.0 / (10000 ** (torch.arange(0, c, 2, device=device).float() / c))
    dtype = inv_freq.dtype

    pos_w = torch.arange(w, device=device, dtype=dtype)
    pos_l = torch.arange(l, device=device, dtype=dtype)
    pos_h = torch.arange(h, device=device, dtype=dtype)
    sin_inp_w = torch.einsum("i,j->ji", pos_w, inv_freq)
    sin_inp_l = torch.einsum("i,j->ji", pos_l, inv_freq)
    sin_inp_h = torch.einsum("i,j->ji", pos_h, inv_freq)
    emb_w = torch.cat((sin_inp_w.sin(), sin_inp_w.cos()), dim=0)[:, :, None, None]
    emb_l = torch.cat((sin_inp_l.sin(), sin_inp_l.cos()), dim=0)[:, None, :, None]
    emb_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=0)[:, None, None, :]
    emb = torch.zeros((1, c*3, w, l, h), device=device, dtype=dtype)
    emb[0, :c, :, :, :] = emb_w
    emb[0, c:2*c, :, :, :] = emb_l
    emb[0, 2*c:, :, :, :] = emb_h
    return emb.repeat((b, 1, 1, 1, 1))
