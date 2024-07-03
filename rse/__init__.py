from torch.nn import functional as F
from functools import partial
from .attacks import *

ATTACKS = (
    'ce-padam', 'cos-padam', 
    'mask-ce-avg-apgd', 'segpgd-loss-apgd', 'js-avg-apgd', 'mask-norm-corrlog-avg-apgd',
    'dag-001', 'dag-003', 'pdpgd', 'alma-prox'
)

@torch.no_grad()
def build_attacks(attack_names, num_classes, eps=8 / 255):
    attacks = {
        'ce-padam': partial(padam, loss_fn=partial(F.cross_entropy, reduction='none'), num_steps=200, step_size=2 / 255, eps=eps),
        'cos-padam': partial(padam, loss_fn=lambda z, y: 1 - F.cosine_similarity(z, F.one_hot(y, num_classes).movedim(-1, 1)), num_steps=200, step_size= 2 / 255, eps=eps),

        'mask-ce-avg-apgd': partial(apgd, n_iter=300, n_restarts=1, use_rs=True, loss='mask-ce-avg', track_loss='ce-avg', eps=eps),
        'segpgd-loss-apgd': partial(apgd, n_iter=300, n_restarts=1, use_rs=True, loss='segpgd-loss', track_loss='ce-avg', eps=eps),
        'js-avg-apgd': partial(apgd, n_iter=300, n_restarts=1, use_rs=True, loss='js-avg', track_loss='ce-avg', eps=eps),
        'mask-norm-corrlog-avg-apgd': partial(apgd, n_iter=300, n_restarts=1, use_rs=True, loss='mask-norm-corrlog-avg', track_loss='mask-norm-corrlog-avg', eps=eps),

        'dag-001': lambda m, i, l: torch.clamp_(dag(m, i, l, max_iter=200, γ=0.001), i - eps, i + eps),
        'dag-003': lambda m, i, l: torch.clamp_(dag(m, i, l, max_iter=200, γ=0.003), i - eps, i + eps),
        'pdpgd': lambda m, i, l: torch.clamp_(pdpgd(m, i, l, norm=float('inf'), primal_lr=0.01), i - eps, i + eps),
        'alma-prox': lambda m, i, l: torch.clamp_(alma_prox(m, i, l, lr_init=0.0001), i - eps, i + eps),
    }
    return [attacks[attack_name] for attack_name in attack_names]

@torch.no_grad()
def seg_attack(model, images, labels, attacks, inv_score_fn=None):
    if inv_score_fn:
        best_images = torch.zeros_like(images)
        best_scores = torch.full((images.size(0),), float('inf'), device=images.device)

        for attack in attacks:
            adv_images = attack(model, images, labels)

            scores = inv_score_fn(model(adv_images), labels)
            replace = scores < best_scores
            replace_view = replace.view(-1, *[1] * (images.dim() - 1))

            best_images = replace_view * adv_images + replace_view.logical_not() * best_images
            best_scores = replace * scores + replace.logical_not() * best_scores
    else:
        best_images = []

        for attack in attacks:
            adv_images = attack(model, images, labels)

            best_images.append(adv_images)

    return best_images

