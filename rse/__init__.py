from torch.nn import functional as F
from functools import partial
from .attacks import *

ATTACKS = {
    'ce-padam': partial(padam, loss_fn=partial(F.cross_entropy, reduction='none'), num_steps=200, step_size=2 / 255),
    'cos-padam': partial(padam, loss_fn=lambda z, y: 1 - F.cosine_similarity(z, y), num_steps=200, step_size= 2 / 255),

    'mask-ce-avg-apgd': partial(apgd_restarts, n_iter=300, n_restarts=1, use_rs=True, loss='mask-ce-avg', track_loss='ce-avg'),
    'segpgd-loss-apgd': partial(apgd_restarts, n_iter=300, n_restarts=1, use_rs=True, loss='segpgd-loss', track_loss='ce-avg'),
    'js-avg-apgd': partial(apgd_restarts, n_iter=300, n_restarts=1, use_rs=True, loss='js-avg', track_loss='ce-avg'),
    'mask-norm-corrlog-avg-apgd': partial(apgd_restarts, n_iter=300, n_restarts=1, use_rs=True, loss='mask-norm-corrlog-avg', track_loss='mask-norm-corrlog-avg'),

    #'dag-001': partial(dag, max_iter=200, γ=0.001),
    #'dag-003': partial(dag, max_iter=200, γ=0.003),
    #'pdpgd': partial(pdpgd, norm=float('inf'), primal_lr=0.01),
    #'alma-prox': partial(alma_prox, lr_init=0.0001)
}

@torch.no_grad()
def build_attacks(attack_names=list(ATTACKS.keys()), eps=8 / 255):
    attacks = []
    for attack_name in attack_names:
        if any(name in attack_name for name in ('dag', 'pdpgd', 'alma-prox')):
            attack = ATTACKS[attack_name]
            attacks.append(lambda m, i, l: torch.clamp_(attack(m, i, l), i - eps, i + eps))
        else:
            attacks.append(partial(ATTACKS[attack_name], eps=eps))
    return attacks

@torch.no_grad()
def seg_attack(model, images, labels, attacks, score_fn=None):
    if score_fn:
        best_images = torch.zeros_like(images)
        best_scores = torch.full((images.size(0),), -float('inf'), device=images.device)

        for attack in attacks:
            adv_images = attack(model, images, labels)

            scores = score_fn(model(adv_images), labels)
            replace = scores > best_scores
            replace_view = replace.view(-1, *[1] * (images.dim() - 1))

            best_images = replace_view * adv_images + replace_view.logical_not() * best_images
            best_scores = replace * scores + replace.logical_not() * best_scores
    else:
        best_images = []

        for attack in attacks:
            adv_images = attack(model, images, labels)

            best_images.append(adv_images)

    return best_images

