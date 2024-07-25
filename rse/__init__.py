from torch.nn import functional as F
from functools import partial
from .attacks import *

ATTACKS = (
    'no-attack',
    'padam-ce', 'padam-cos', 
    'apgd-mask-ce-avg', 'apgd-segpgd-loss', 'apgd-js-avg', 'apgd-mask-norm-corrlog-avg',
    'dag-001', 'dag-003', 'pdpgd', 'alma-prox'
)

def build_attacks(attack_names, num_classes, eps=8 / 255):
    attacks = {
        'no-attack': lambda m, i, l: i,

        'padam-ce': partial(padam, loss_fn=partial(F.cross_entropy, reduction='none'), num_steps=200, step_size=2 / 255, eps=eps),
        'padam-cos': partial(padam, loss_fn=lambda z, y: 1 - F.cosine_similarity(z, F.one_hot(y, num_classes).movedim(-1, 1)), num_steps=200, step_size= 2 / 255, eps=eps),

        'apgd-mask-ce-avg': partial(apgd, n_iter=300, n_restarts=1, use_rs=True, loss='mask-ce-avg', track_loss='ce-avg', eps=eps),
        'apgd-segpgd-loss': partial(apgd, n_iter=300, n_restarts=1, use_rs=True, loss='segpgd-loss', track_loss='ce-avg', eps=eps),
        'apgd-js-avg': partial(apgd, n_iter=300, n_restarts=1, use_rs=True, loss='js-avg', track_loss='ce-avg', eps=eps),
        'apgd-mask-norm-corrlog-avg': partial(apgd, n_iter=300, n_restarts=1, use_rs=True, loss='mask-norm-corrlog-avg', track_loss='mask-norm-corrlog-avg', eps=eps),

        'dag-001': lambda m, i, l: torch.clamp_(dag(m, i, l, max_iter=200, γ=0.001), i - eps, i + eps),
        'dag-003': lambda m, i, l: torch.clamp_(dag(m, i, l, max_iter=200, γ=0.003), i - eps, i + eps),
        'pdpgd': lambda m, i, l: torch.clamp_(pdpgd(m, i, l, norm=float('inf'), primal_lr=0.01), i - eps, i + eps),
        'alma-prox': lambda m, i, l: torch.clamp_(alma_prox(m, i, l, lr_init=0.0001), i - eps, i + eps),
    }
    return [attacks[attack_name] for attack_name in attack_names]

@torch.no_grad()
def composite_attack(model, images, labels, attacks, inv_score_fn=None):
    if inv_score_fn:
        best_images = torch.zeros_like(images)
        best_scores = torch.full((images.size(0),), float('inf'), device=images.device)

        for attack in attacks:
            with torch.enable_grad():
                adv_images = attack(model, images, labels)

            scores = inv_score_fn(model(adv_images).cpu(), labels.cpu())
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

@torch.no_grad()
def evaluate(model, dataloader, attack, metrics, device=None, dtype=None, max_samples=float('inf')):
    sample_params = next(model.parameters())
    if not device:
        device = sample_params.device
    if not dtype:
        dtype = sample_params.dtype

    model.eval()

    for m in metrics:
        m.reset()

    evaluated_samples = 0
    for images, labels in dataloader:
        num_samples = images.size(0)
        samples_to_eval = min(max_samples - evaluated_samples, num_samples)

        images = images[:samples_to_eval].to(device=device, dtype=dtype, non_blocking=True)
        labels = labels[:samples_to_eval].to(device=device, non_blocking=True)

        with torch.enable_grad():
            adv_images = attack(model, images, labels)
        preds = model(adv_images)

        for m in metrics:
            m.update(preds.cpu(), labels.cpu())

        evaluated_samples += samples_to_eval
        if evaluated_samples == max_samples:
            break

    results = [m.compute().item() for m in metrics]

    for m in metrics:
        m.reset()

    return results

