"""
utils.py — Utility functions for S-FSCIL
  - Herding-based exemplar selection
  - CLIP-guided pseudo-label selection
  - Metric computation
  - Logging helpers
  - Reproducibility
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Herding-based exemplar selection ─────────────────────────────────────────

@torch.no_grad()
def herding_select(model, dataset, class_ids, budget_per_class, device):
    """
    Select budget_per_class exemplars per class using herding.
    Exemplar = sample closest to class prototype in feature space.

    Args:
        model:            SFSCILModel (backbone used for feature extraction)
        dataset:          Dataset for this session (labeled)
        class_ids:        list of class indices in this session
        budget_per_class: M (default 1)
        device:           torch device

    Returns:
        exemplars: list of (img_tensor, label) tuples
    """
    model.eval()
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Collect features and labels
    features, labels = [], []
    for imgs, lbls in loader:
        phi = model.encode(imgs.to(device))
        features.append(phi.cpu())
        labels.append(lbls)
    features = torch.cat(features, dim=0)   # [N, d]
    labels   = torch.cat(labels, dim=0)     # [N]

    exemplars = []
    for cls in class_ids:
        mask = (labels == cls)
        cls_feats = features[mask]           # [n_cls, d]
        cls_imgs  = [dataset[i][0]
                     for i in range(len(dataset)) if labels[i] == cls]

        # Class prototype μ_c
        mu_c = cls_feats.mean(dim=0, keepdim=True)   # [1, d]

        # Euclidean distances d(x_j, c) = || μ_c - φ(x_j) ||_2
        dists = torch.cdist(mu_c, cls_feats).squeeze(0)  # [n_cls]

        # Select top-M closest
        _, top_idx = torch.topk(dists, k=min(budget_per_class, len(dists)),
                                largest=False)
        for idx in top_idx.tolist():
            exemplars.append((cls_imgs[idx], cls))

    return exemplars


# ── CLIP-guided pseudo-label selection ───────────────────────────────────────

@torch.no_grad()
def select_pseudo_labels(model, unlabeled_pool, session_id, args, device):
    """
    Two-stage pseudo-label selection:
      Stage 1: confidence gate  — retain u_j if max(q_j) >= τ_c
      Stage 2: pool ranking     — rank eligible samples by r_j = max(q_j ⊙ s_j)
                                   (unnormalised joint score; not a probability)
      Select top-S per class independently to enforce class balance.

    Args:
        model:          SFSCILModel
        unlabeled_pool: list of (img_tensor, true_label) [true labels UNSEEN]
        session_id:     int
        args:           parsed config
        device:         torch device

    Returns:
        selected: list of (img_tensor, pseudo_label) for selected samples
    """
    model.eval()
    S = max(1, int(args.unlabeled_pool * args.selection_ratio))
    tau_c = args.tau_c
    tau   = args.tau_clip

    # ── Forward pass: model predictions (weakly augmented view) ──────────────
    from datasets import get_weak_augmentation, get_img_size
    from torch.utils.data import DataLoader, TensorDataset
    img_size = get_img_size(args.dataset)
    weak_tf = get_weak_augmentation(args.dataset, img_size)

    pool_imgs_orig = [item[0] for item in unlabeled_pool]  # original PIL/tensor

    # Model confidence q_j on weakly augmented view
    q_list, z_clip_list, img_list = [], [], []
    for img_orig in pool_imgs_orig:
        img_w = weak_tf(img_orig).unsqueeze(0).to(device)
        phi   = model.encode(img_w)
        q_j   = torch.softmax(model.classify(phi), dim=-1).squeeze(0)  # [C]

        # CLIP similarity on ORIGINAL (non-augmented) image
        z_j   = model.clip_similarity(img_w, session_id).squeeze(0)    # [C]
        # Softmax with temperature τ to get s_j
        s_j   = torch.softmax(z_j / tau, dim=-1)

        q_list.append(q_j.cpu())
        z_clip_list.append(s_j.cpu())
        img_list.append(img_orig)

    q_all      = torch.stack(q_list)      # [N, C]
    s_all      = torch.stack(z_clip_list) # [N, C]

    # ── Stage 1: confidence gate on max(q_j) — calibrated probability ─────
    max_q      = q_all.max(dim=-1).values  # [N]
    eligible   = (max_q >= tau_c).nonzero(as_tuple=True)[0]  # indices

    # ── Refined pseudo-label: q_hat_j = argmax(q_j ⊙ s_j) ────────────────
    q_elig     = q_all[eligible]           # [n_elig, C]
    s_elig     = s_all[eligible]           # [n_elig, C]
    joint      = q_elig * s_elig           # element-wise; unnormalised
    pseudo_lbl = joint.argmax(dim=-1)      # [n_elig]

    # ── Stage 2: rank by r_j = max(q_j ⊙ s_j) — per class ───────────────
    r_j        = joint.max(dim=-1).values  # [n_elig] unnormalised ranking score

    # Group by predicted pseudo-label
    class_buckets = {}
    for i, (orig_idx, plbl) in enumerate(
            zip(eligible.tolist(), pseudo_lbl.tolist())):
        if plbl not in class_buckets:
            class_buckets[plbl] = []
        class_buckets[plbl].append((r_j[i].item(), orig_idx, plbl))

    # Select top-S per class; break ties by Euclidean distance to prototype
    selected = []
    for cls, candidates in class_buckets.items():
        candidates.sort(key=lambda x: -x[0])  # descending r_j
        top = candidates[:S]
        for _, orig_idx, plbl in top:
            selected.append((img_list[orig_idx], plbl))

    return selected


# ── Loss functions ────────────────────────────────────────────────────────────

def distillation_loss(z_hat, z_t, old_class_indices, T=2.0):
    """
    Semantic-aware distillation loss restricted to C^(t-1).
    Softmax denominator also restricted to C^(t-1) — not C^(t).

    L_dis = Σ_{c ∈ C^(t-1)} -σ_c^old(ẑ) log σ_c^old(z_t)

    Args:
        z_hat:             [B, C^(t)]  fused supervisory target
        z_t:               [B, C^(t)]  current model logits
        old_class_indices: list of int indices for C^(t-1)
        T:                 distillation temperature

    Returns:
        scalar loss
    """
    # Restrict to old classes
    z_hat_old = z_hat[:, old_class_indices]   # [B, |C^(t-1)|]
    z_t_old   = z_t[:, old_class_indices]     # [B, |C^(t-1)|]

    # Temperature-scaled softmax over C^(t-1) only
    p_teacher = F.softmax(z_hat_old / T, dim=-1)
    p_student = F.log_softmax(z_t_old / T, dim=-1)

    return -(p_teacher * p_student).sum(dim=-1).mean()


def consistency_loss(model, x, eta, eta_prime):
    """
    L_c = (1/B_l) Σ || p_θ(y|η(x)) - p_θ(y|η'(x)) ||_2^2
    eta, eta_prime: independently perturbed views of x
    """
    p1 = torch.softmax(model.classify(model.encode(eta)), dim=-1)
    p2 = torch.softmax(model.classify(model.encode(eta_prime)), dim=-1)
    return F.mse_loss(p1, p2)


# ── Accuracy computation ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(model, loader, device):
    """Top-1 accuracy over all classes in loader."""
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        phi = model.encode(imgs)
        logits = model.classify(phi)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ── Model snapshot (frozen teacher copies) ───────────────────────────────────

def snapshot_model(model):
    """Return a frozen deep copy of model for use as teacher."""
    m = deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


# ── Logging ───────────────────────────────────────────────────────────────────

class SessionLogger:
    """Tracks per-session accuracy and computes average + improvement."""

    def __init__(self, base_session_baseline=None):
        self.session_accs = []
        self.baseline = base_session_baseline  # for Imp↑ computation

    def log(self, session_id, acc):
        self.session_accs.append(acc)
        print(f"  Session {session_id}: {acc:.2f}%")

    def summary(self):
        avg = np.mean(self.session_accs)
        print(f"\n  Average: {avg:.2f}%")
        if self.baseline is not None:
            print(f"  Imp↑:    {avg - self.baseline:.2f}%")
        return avg

    def to_dict(self):
        return {f"s{i}": acc for i, acc in enumerate(self.session_accs)}
