import torch
from torch import optim as optim


def calc_bert_lr(lr, gstep, warmup_steps, hold_steps, cooldown_steps):
    if gstep < warmup_steps:
        factor = gstep / warmup_steps
    elif warmup_steps <= gstep < warmup_steps + hold_steps:
        factor = 1.0
    elif warmup_steps + hold_steps <= gstep < warmup_steps + hold_steps + cooldown_steps:
        factor = (warmup_steps + hold_steps + cooldown_steps - gstep) / cooldown_steps
    else:
        factor = 0
    lro = lr * factor
    return lro


def switch_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_eval_loop(dataloader,
                    model,
                    writer,
                    val,
                    optimargs=None,
                    gstep=0,
                    device="cpu",
                    optimizers=None,
                    optimizer_states=None):
    prefix = "val" if val else "train"

    if not val:
        # -------------------------------------------------------------------------------------
        # OPTIMIZER SETUP
        if optimargs is None:
            optimargs = {"lr": 0.001, "weight_decay": 1e-8}

        bert_lr = optimargs["bert"].lr
        bert_warmup_steps = optimargs["bert"].warmup_steps
        bert_hold_steps = optimargs["bert"].hold_steps
        bert_cooldown_steps = optimargs["bert"].cooldown_steps

        # Create optimizers if not already supplied
        if optimizers is None:
            all_params = {k: v for k, v in model.named_parameters()}
            bert_params = {k: v for k, v in all_params.items() if "bertmodel" in k}
            nonbert_params = {k: v for k, v in all_params.items() if "bertmodel" not in k}

            nonbert_optimizer = optim.Adam(nonbert_params.values(),
                                           lr=optimargs["nonbert"].lr,
                                           weight_decay=optimargs["nonbert"].weight_decay)
            if len(bert_params) > 0:
                bert_optimizer = optim.Adam(bert_params.values(),
                                            lr=optimargs["bert"].lr,
                                            weight_decay=optimargs["bert"].weight_decay)
            else:
                bert_optimizer = None

            # Initilize optimizers from provided states
            if optimizer_states is not None:
                nonbert_optimizer.load_state_dict(optimizer_states[0])
                if bert_optimizer is not None:
                    bert_optimizer.load_state_dict(optimizer_states[1])
        else:
            nonbert_optimizer, bert_optimizer = optimizers
    else:
        nonbert_optimizer, bert_optimizer = None, None

    # -------------------------------------------------------------------------------------
    # LOOP

    for i, batch in enumerate(dataloader):

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        loss, metrics = model(batch)
        loss = loss.mean()

        if not val:
            gstep += 1
            if bert_optimizer is not None:
                bert_optimizer.zero_grad()

            nonbert_optimizer.zero_grad()
            loss.backward()
            nonbert_optimizer.step()

            if bert_optimizer is not None:
                bert_optimizer.step()
                bert_step_lr = calc_bert_lr(bert_lr, gstep, bert_warmup_steps, bert_hold_steps, bert_cooldown_steps)
                switch_lr(bert_optimizer, bert_step_lr)
                metrics["bert_step_lr"] = bert_step_lr

        print(f"Iter: {i}, " + " | ".join([f"{k}: {v}" for k, v in metrics.items()]))
        if writer is not None:
            writer.add_scalar_dict(f"{prefix}/rewardvalue", metrics)
            writer.inc_iter()

    return gstep, (nonbert_optimizer, bert_optimizer)