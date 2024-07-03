import torch
from ._sea import apgd_restarts as apgd
from adv_lib.attacks.segmentation import alma_prox, pdpgd, dag


@torch.no_grad()
def padam(model, x, y, loss_fn, eps, step_size, num_steps, bounds=(0, 1), random_start=True, targeted=False):
    multiplier = -1.0 if targeted else 1.0
    delta = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    if random_start:
        delta = delta.uniform_(-eps, eps)
        delta = (x + delta).clamp(*bounds) - x
    adam = torch.optim.Adam([delta], lr=step_size, amsgrad=True, maximize=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            delta.requires_grad = True
            loss = multiplier * loss_fn(model(x + delta), y).mean()
            delta.grad = torch.autograd.grad(loss, delta)[0]
            adam.step()
        delta.clamp_(-eps, eps)
        delta.copy_((x + delta).clamp(*bounds) - x, non_blocking=True)
    return x + delta

dag = torch.enable_grad(dag)
pdpgd = torch.enable_grad(pdpgd)
alma_prox = torch.enable_grad(alma_prox)

