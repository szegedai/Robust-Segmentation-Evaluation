# Francesco Croce, Naman D Singh, Matthias Hein
# robust-segmentation
# https://github.com/nmndeep/robust-segmentation/blob/main/semseg/utils/attacker.py
# This is a modified version of the original file.

import torch
import torch.nn.functional as F
from functools import partial


def dlr_loss(x, y, reduction='none'):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
        
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
        x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
        x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)


def cospgd_loss(pred, target, reduction='mean', ignore_index=-1):
    """Implementation of the loss for semantic segmentation from
    https://arxiv.org/abs/2302.02213.

    pred: B x cls x h x w
    target: B x h x w
    """

    sigm_pred = torch.sigmoid(pred)
    sh = target.shape
    n_cls = pred.shape[1]
    
    mask_background = (target != ignore_index).long()
    y = mask_background * target  # One-hot encoding doesn't support -1.
    y = F.one_hot(y.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    w = F.cosine_similarity(sigm_pred, y)
    w = mask_background * w  # Ignore pixels with label -1.
    
    loss = F.cross_entropy(
        pred, target, reduction='none', ignore_index=ignore_index)
    assert w.shape == loss.shape
    loss = w.detach() * loss

    if reduction == 'mean':
        return loss.view(sh[0], -1).mean(-1)

    return loss


def masked_cross_entropy(pred, target, reduction='none', ignore_index=-1):
    """Cross-entropy of only correctly classified pixels."""

    mask = pred.max(1)[1] == target
    mask = (target != ignore_index) * mask
    loss = F.cross_entropy(pred, target, reduction='none', ignore_index=-1)
    loss = mask.float().detach() * loss
    
    if reduction == 'mean':
        return loss.view(pred.shape[0], -1).mean(-1)
    return loss


def margin_loss(pred, target):

    sh = target.shape
    n_cls = pred.shape[1]
    y = F.one_hot(target.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    logits_target = (y * pred).sum(1)
    logits_other = (pred - 1e10 * y).max(1)[0]

    return logits_other - logits_target


def masked_margin_loss(pred, target):
    """Margin loss of only correctly classified pixels."""

    pred = pred / (pred ** 2).sum(1, keepdim=True).sqrt().detach() 
    loss = margin_loss(pred, target)
    mask = pred.max(1)[1] == target
    loss = mask.float().detach() * loss 

    return loss.view(pred.shape[0], -1).mean(-1)


def single_logits_loss(pred, target, normalized=False, reduction='none',
        masked=False, ignore_index=-1):
    """The (normalized) logit of the correct class is minimized."""

    if normalized:
        pred = pred / (pred ** 2).sum(1, keepdim=True).sqrt() 
    sh = target.shape
    n_cls = pred.shape[1]
    mask_background = (target != ignore_index).long()
    y = target * mask_background  # One-hot doesn't support -1 class.
    y = F.one_hot(y.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    loss = -1 * (y * pred).sum(1)
    loss = loss * mask_background  # Ignore contribution of background.
    if masked:
        mask = pred.max(1)[1] == target
        loss = mask.float().detach() * loss

    if reduction == 'mean':
        return loss.view(sh[0], -1).mean(-1)
    return loss


def targeted_single_logits_loss(
    pred, labels, target, normalized=False, reduction='none',
    masked=False):
    """The (normalized) logit of the target class is maximized."""

    if normalized:
        pred = pred / (pred ** 2).sum(1, keepdim=True).sqrt() #.detach()
    sh = target.shape
    n_cls = pred.shape[1]
    y = F.one_hot(target.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    loss = (y * pred).sum(1)
    if masked:
        mask = pred.max(1)[1] == labels
        loss = mask.float().detach() * loss

    if reduction == 'mean':
        return loss.view(sh[0], -1).mean(-1)
    return loss


def js_div_fn(p, q, softmax_output=False, reduction='none', red_dim=None,
    ignore_index=-1):
    """Compute JS divergence between p and q.

    p: logits [bs, n_cls, ...]
    q: labels [bs, ...]
    softmax_output: if softmax has already been applied to p
    reduction: to pass to KL computation
    red_dim: dimensions over which taking the sum
    ignore_index: the pixels with this label are ignored
    """
    
    if not softmax_output:
        p = F.softmax(p, 1)
    mask_background = (q != ignore_index).long()
    if reduction != 'none' and mask_background.sum() > 0:
        raise ValueError('Incompatible setup.')
    q = mask_background * q  # Change labels -1 to 0 for one-hot.
    q = F.one_hot(q.view(q.shape[0], -1), p.shape[1])
    q = q.permute(0, 2, 1).view(p.shape).float()
    
    m = (p + q) / 2
    
    loss = (F.kl_div(m.log(), p, reduction=reduction)
            + F.kl_div(m.log(), q, reduction=reduction)) / 2
    loss = mask_background.unsqueeze(1) * loss  # Ignore contribution of background.
    if red_dim is not None:
        assert reduction == 'none', 'Incompatible setup.'
        loss = loss.sum(dim=red_dim)
    
    return loss


def js_loss(p, q, reduction='mean'):

    loss = js_div_fn(p, q, red_dim=(1))  # Sum over classes.
    if reduction == 'mean':
        return loss.view(p.shape[0], -1).mean(-1)
    elif reduction == 'none':
        return loss


def segpgd_loss(pred, target, t, max_t, reduction='none', ignore_index=-1):
    """Implementation of the loss of https://arxiv.org/abs/2207.12391.

    pred: B x cls x h x w
    target: B x h x w
    t: current iteration
    max_t: total iterations
    """

    lmbd = t / 2 / max_t
    corrcl = (pred.max(1)[1] == target).float().detach()
    loss = F.cross_entropy(pred, target, reduction='none',
        ignore_index=ignore_index)
    loss = (1 - lmbd) * corrcl * loss + lmbd * (1 - corrcl) * loss

    if reduction == 'mean':
        return loss.view(target.shape[0], -1).mean(-1)
    return loss


def pixel_to_img_loss(loss, mask_background=None):

    if mask_background is not None:
        loss = mask_background * loss
    return loss.view(loss.shape[0], -1).mean(-1)


def check_oscillation(x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(x.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()


criterion_dict = {
    'ce': lambda x, y: F.cross_entropy(
        x, y, reduction='none', ignore_index=-1),
    'dlr': dlr_loss,
    'dlr-targeted': dlr_loss_targeted,
    'ce-avg': lambda x, y: F.cross_entropy(
        x, y, reduction='none', ignore_index=-1),
    'cospgd-loss': partial(cospgd_loss, reduction='none'),
    'mask-ce-avg': masked_cross_entropy,
    'margin-avg': lambda x, y: margin_loss(x, y).view(x.shape[0], -1).mean(-1),
    'mask-margin-avg': masked_margin_loss,
    'js-avg': partial(js_loss, reduction='none'),
    'segpgd-loss': partial(segpgd_loss, reduction='none'),
    'mask-norm-corrlog-avg': partial(
        single_logits_loss, normalized=True, reduction='none', masked=True),
    'mask-norm-corrlog-avg-targeted': partial(
        targeted_single_logits_loss, normalized=True, reduction='none',
        masked=True),
    }


def apgd_train(model, x, y, eps, n_iter=10, use_rs=False, loss='ce', track_loss=None, y_target=None, ignore_index=-1, x_init=None):
    assert not model.training
    assert ignore_index == -1, 'Only `ignore_index = 1` is supported.'
    device = x.device
    ndims = len(x.shape) - 1
    bs = x.shape[0]
    loss_name = loss

    if not use_rs:
        x_adv = x.clone()
    else:
        t = 2 * torch.rand_like(x) - 1
        x_adv = (x.clone() + eps * t).clamp(0., 1.)
    if x_init is not None:
        x_adv = x_init.clone()

    # Set mask for background pixels: this might not be needed for losses
    # which incorporate `ignore_index` already (e.g. CE), but shouldn't
    # influence the results.
    mask_background = y == ignore_index
    mask_background = 1 - mask_background.float()
    
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)
    
    # set loss
    criterion_indiv = criterion_dict[loss]
    if track_loss is None:
        track_loss = loss
    track_loss_fn = criterion_dict[track_loss]

    # set params
    n_iter_2 = max(int(0.22 * n_iter), 1)
    n_iter_min = max(int(0.06 * n_iter), 1)
    size_decr = max(int(0.03 * n_iter), 1)
    k = n_iter_2 + 0
    thr_decr = .75
    alpha = 2.
    
    step_size = alpha * eps * torch.ones([x.shape[0], *[1] * ndims], device=device)
    counter3 = 0

    x_adv.requires_grad_()
    logits = model(x_adv)
    if loss_name not in ['orth-ce-avg']:
        if loss_name == 'segpgd-loss':
            loss_indiv = criterion_indiv(logits, y, 0, n_iter)
        elif loss_name in [
            'mask-norm-corrlog-avg-targeted', 'norm-corrlog-avg-targeted']:
            loss_indiv = criterion_indiv(logits, y, y_target)
        else:
            loss_indiv = criterion_indiv(logits, y)
        loss_indiv = pixel_to_img_loss(loss_indiv, mask_background)
        loss = loss_indiv.sum()
        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        # To potentially use a different loss e.g. no mask.
        if track_loss == 'segpgd-loss':
            loss_indiv = track_loss_fn(logits, y, 0, n_iter)
        elif track_loss in [
            'mask-norm-corrlog-avg-targeted', 'norm-corrlog-avg-targeted']:
            loss_indiv = track_loss_fn(logits, y, y_target)
        else:
            loss_indiv = track_loss_fn(logits, y)
        loss_indiv = pixel_to_img_loss(loss_indiv, mask_background)
        loss = loss_indiv.sum()
    else:
        loss_indiv, grad = criterion_indiv(logits, y, x_adv)
        loss = loss_indiv.sum()
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()

    acc = logits.detach().max(1)[1] == y
    acc = acc.float().view(bs, -1).mean(-1)
    acc_steps[0] = acc + 0
    pred_best = logits.detach().max(1)[1]  # Track the predictions for best points.
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    
    x_adv_old = x_adv.clone().detach()
    
    for i in range(n_iter):
        ### gradient step
        x_adv = x_adv.detach()
        grad2 = x_adv - x_adv_old
        x_adv_old = x_adv.clone()
            
        a = 0.75 if i > 0 else 1.0

        x_adv_1 = x_adv + step_size * torch.sign(grad)
        x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - eps), x + eps), 0.0, 1.0)
        x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - eps), x + eps), 0.0, 1.0)
                
        x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        logits = model(x_adv)
        if loss_name not in ['orth-ce-avg']:
            if loss_name == 'segpgd-loss':
                loss_indiv = criterion_indiv(logits, y, i + 1, n_iter)
            elif loss_name in [
                'mask-norm-corrlog-avg-targeted', 'norm-corrlog-avg-targeted']:
                loss_indiv = criterion_indiv(logits, y, y_target)
            else:
                loss_indiv = criterion_indiv(logits, y)
            loss_indiv = pixel_to_img_loss(loss_indiv, mask_background)
            loss = loss_indiv.sum()
            
            if i < n_iter - 1:
                # save one backward pass
                grad = torch.autograd.grad(loss, [x_adv])[0].detach()

            # To potentially use a different loss e.g. no mask.
            if track_loss == 'segpgd-loss':
                loss_indiv = track_loss_fn(logits, y, i + 1, n_iter)
            elif track_loss in [
                'mask-norm-corrlog-avg-targeted', 'norm-corrlog-avg-targeted']:
                loss_indiv = track_loss_fn(logits, y, y_target)
            else:
                loss_indiv = track_loss_fn(logits, y)
            loss_indiv = pixel_to_img_loss(loss_indiv, mask_background)
            loss = loss_indiv.sum()
        else:
            loss_indiv, grad = criterion_indiv(logits, y, x_adv)
            loss = loss_indiv.sum()
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()
        
        # Save images with lowest average accuracy over pixels.
        pred = logits.detach().max(1)[1] == y
        # Set pixels of the background to be correctly classified: this
        # overestimates accuracy but doesn't change the order, hence shouldn't
        # influence the results.
        pred[y == ignore_index] = True
        avg_acc = pred.float().view(bs, -1).mean(-1)
        ind_pred = (avg_acc <= acc).nonzero().squeeze()
        acc = torch.min(acc, avg_acc)
        acc_steps[i + 1] = acc + 0
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        pred_best[ind_pred] = logits.detach().max(1)[1][ind_pred] + 0

        ### check step size
        y1 = loss_indiv.detach().clone()
        loss_steps[i] = y1 + 0
        ind = (y1 > loss_best).nonzero().squeeze()
        x_best[ind] = x_adv[ind].clone()
        grad_best[ind] = grad[ind].clone()
        loss_best[ind] = y1[ind] + 0
        loss_best_steps[i + 1] = loss_best + 0

        counter3 += 1

        if counter3 == k:
            fl_oscillation = check_oscillation(loss_steps, i, k, loss_best, k3=thr_decr)
            fl_reduce_no_impr = (1. - reduced_last_check) * (loss_best_last_check >= loss_best).float()
            fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
            reduced_last_check = fl_oscillation.clone()
            loss_best_last_check = loss_best.clone()

            if fl_oscillation.sum() > 0:
                ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                step_size[ind_fl_osc] /= 2.0

                x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                  
            counter3 = 0
            k = max(k - size_decr, n_iter_min)
              
    return x_best_adv


@torch.enable_grad()
def apgd_restarts(model, x, y, eps=8. / 255., n_iter=10, loss='ce', n_restarts=1, track_loss=None, use_rs=False, ignore_index=-1):
    """Run apgd with the option of restarts."""

    acc = torch.ones([x.shape[0]], device=x.device) # run on all points
    x_adv = x.clone()
    y_target = None
    if 'targeted' in loss:
        with torch.no_grad():
            output = model(x)
        outputsorted = output.sort(1)[1]
        n_target_classes = 21 # max number of target classes to use
    
    for i in range(n_restarts):
        ind = acc > 0
        if acc.sum() > 0:
            if 'targeted' in loss:
                target_cls = i % n_target_classes + 1
                y_target = outputsorted[:, -target_cls].clone()
                mask = (y_target == y).long()
                other_target = (
                    outputsorted[:, -target_cls - 1] if i == 0 else
                    outputsorted[:, -target_cls + 1])
                y_target = y_target * (1 - mask) + other_target * mask
            
            x_adv_curr = apgd_train(
                model, x[ind], y[ind],
                n_iter=n_iter, use_rs=use_rs, loss=loss,
                eps=eps, track_loss=track_loss,
                y_target=y_target[ind] if y_target is not None else None,
                ignore_index=ignore_index,
            )
            
            with torch.no_grad():
                pred = model(x_adv_curr).max(1)[1] == y[ind]
            pred[y[ind] == ignore_index] = True
            acc_curr = pred.float().view(x_adv_curr.shape[0], -1).mean(-1)
            to_update = acc_curr < acc[ind]
            succs = torch.nonzero(ind).squeeze()
            if len(succs.shape) == 0:
                succs.unsqueeze_(0)
            x_adv[succs[to_update]] = x_adv_curr[to_update].clone()
            acc[succs[to_update]] = acc_curr[to_update].clone()
            
    return x_adv

