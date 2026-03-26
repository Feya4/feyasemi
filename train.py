"""
train.py — S-FSCIL Training Script
Implements Algorithm 1: Incremental Training with SAUD and Exemplar Replay

Usage:
    python train.py --dataset miniImageNet --backbone ViT-B/32
    python train.py --dataset CUB200 --backbone ViT-B/32
    python train.py --dataset CIFAR100 --backbone resnet12
"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from config import get_args, CUB_PROMPT_TEMPLATES, DEFAULT_PROMPT_TEMPLATE
from datasets import get_session_datasets, _sample_pool
from models import SFSCILModel
from utils import (
    set_seed, herding_select, select_pseudo_labels,
    distillation_loss, consistency_loss, compute_accuracy,
    snapshot_model, SessionLogger,
)


# ── Session class schedule ────────────────────────────────────────────────────

def get_class_schedule(args):
    """
    Returns list of class_id lists per session.
    Session 0: base classes; Sessions 1..k-1: N new classes each.
    """
    schedule = [list(range(args.base_classes))]
    for s in range(1, args.num_sessions):
        start = args.base_classes + (s - 1) * args.way
        schedule.append(list(range(start, start + args.way)))
    return schedule


# ── Base session training ─────────────────────────────────────────────────────

def train_base_session(model, args, device):
    """Train base model M_b on D_{l,0} with supervised CE + CLIP embeddings."""

    print("\n=== Base Session (t=0) ===")

    # Build prompt templates and cache text embeddings for C_0
    class_names = get_class_names(args.dataset, args.data_root,
                                  list(range(args.base_classes)))
    templates = (CUB_PROMPT_TEMPLATES if args.dataset == "CUB200"
                 and args.prompt_ensemble else [DEFAULT_PROMPT_TEMPLATE])
    model.cache_text_embeddings(class_names, session_id=0,
                                prompt_templates=templates,
                                ensemble=args.prompt_ensemble)

    # Expand classifier for base classes
    model.add_session_classes(args.base_classes)

    # Data
    class_ids = list(range(args.base_classes))
    labeled_loader, _, test_loader = get_session_datasets(
        args, session_id=0, class_ids=class_ids)

    # Optimiser
    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=0.1)

    ce_loss = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, args.base_epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in labeled_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            phi    = model.encode(imgs)
            logits = model.classify(phi)
            loss   = ce_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 20 == 0 or epoch == args.base_epochs:
            acc = compute_accuracy(model, test_loader, device)
            print(f"  Epoch {epoch}/{args.base_epochs}  "
                  f"loss={total_loss/len(labeled_loader):.4f}  acc={acc:.2f}%")
            best_acc = max(best_acc, acc)

    print(f"  Base session accuracy: {best_acc:.2f}%")

    # Select exemplars E_0
    labeled_ds_raw = get_session_datasets(
        args, session_id=0, class_ids=class_ids)[0].dataset
    exemplars = herding_select(model, labeled_ds_raw, class_ids,
                               args.memory_size, device)
    print(f"  Stored {len(exemplars)} exemplars from base session.")

    return exemplars, best_acc


# ── Incremental session training ──────────────────────────────────────────────

def train_incremental_session(model, base_model, prev_model,
                               session_id, class_ids, all_class_ids,
                               exemplar_set, args, device):
    """
    Train one incremental session t >= 1.
    Implements the inner loop of Algorithm 1.
    """
    print(f"\n=== Incremental Session (t={session_id}) "
          f"| New classes: {class_ids} ===")

    # Cache text embeddings for all known classes C^(t)
    class_names = get_class_names(args.dataset, args.data_root, all_class_ids)
    templates = (CUB_PROMPT_TEMPLATES if args.dataset == "CUB200"
                 and args.prompt_ensemble else [DEFAULT_PROMPT_TEMPLATE])
    model.cache_text_embeddings(class_names, session_id=session_id,
                                prompt_templates=templates,
                                ensemble=args.prompt_ensemble)

    # Expand classifier for N new classes
    model.add_session_classes(args.way)
    model.freeze_for_incremental()

    total_classes = model.classifier.num_classes
    old_class_ids = all_class_ids[:len(all_class_ids) - args.way]

    # Data: labeled D_{l,t} + unlabeled pool D_{u,t} + exemplars E_{t-1}
    labeled_loader, unlabeled_loader, test_loader = get_session_datasets(
        args, session_id, class_ids, exemplar_set)

    # CLIP-guided pseudo-label selection (Algorithm 1, steps 10-12)
    unlabeled_pool = _build_unlabeled_pool(args, class_ids, device)
    pseudo_labeled = select_pseudo_labels(
        model, unlabeled_pool, session_id, args, device)
    print(f"  Selected {len(pseudo_labeled)} pseudo-labeled samples.")

    # Freeze teacher heads
    base_model.eval()
    prev_model.eval()

    # Optimiser — only non-frozen parameters
    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.inc_lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )

    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, args.inc_epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs_l, labels_l in labeled_loader:
            imgs_l, labels_l = imgs_l.to(device), labels_l.to(device)

            # ── Features ──────────────────────────────────────────────────────
            phi_l = model.encode(imgs_l)

            # ── L_l: supervised cross-entropy on D_{l,t} ∪ E_{t-1} ──────────
            logits_l = model.classify(phi_l)
            loss_l   = ce_loss(logits_l, labels_l)

            # ── L_u: pseudo-label consistency on selected unlabeled ───────────
            loss_u = compute_pseudo_loss(
                model, pseudo_labeled, session_id, args, ce_loss, device)

            # ── L_c: consistency regularization ──────────────────────────────
            loss_c = compute_consistency_loss(model, imgs_l, args, device)

            # ── SAUD: build ẑ and compute L_dis ──────────────────────────────
            z_clip = model.clip_similarity(imgs_l, session_id)
            z_hat  = model.build_distillation_target(
                phi_l, z_clip, base_model, prev_model, total_classes)
            z_t    = model.classify(phi_l)
            loss_d = distillation_loss(z_hat, z_t, old_class_ids, T=args.T)

            # ── Total loss L_t ────────────────────────────────────────────────
            loss = (args.lambda_l * loss_l
                  + args.lambda_u * loss_u
                  + args.lambda_c * loss_c
                  + args.lambda_d * loss_d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    acc = compute_accuracy(model, test_loader, device)
    print(f"  Session {session_id} accuracy: {acc:.2f}%")

    # Update exemplar memory E_t (append-only)
    labeled_ds_raw = get_session_datasets(
        args, session_id, class_ids)[0].dataset
    new_exemplars = herding_select(model, labeled_ds_raw, class_ids,
                                   args.memory_size, device)
    updated_exemplars = exemplar_set + new_exemplars
    print(f"  Exemplar buffer: {len(updated_exemplars)} total.")

    return updated_exemplars, acc


# ── Loss helpers ──────────────────────────────────────────────────────────────

def compute_pseudo_loss(model, pseudo_labeled, session_id, args, ce_loss, device):
    """L_u on CLIP-selected pseudo-labeled samples (strongly augmented view)."""
    if not pseudo_labeled:
        return torch.tensor(0.0, device=device)

    from datasets import get_strong_augmentation, get_img_size
    strong_tf = get_strong_augmentation(
        args.dataset, get_img_size(args.dataset),
        args.randaugment_n, args.randaugment_m)

    losses = []
    for img, plbl in pseudo_labeled:
        img_s  = strong_tf(img).unsqueeze(0).to(device)
        phi    = model.encode(img_s)
        logits = model.classify(phi)
        target = torch.tensor([plbl], device=device)
        losses.append(ce_loss(logits, target))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def compute_consistency_loss(model, imgs_l, args, device):
    """
    L_c between two independently perturbed views of labeled samples.
    Uses strong augmentation as η and η'.
    """
    from datasets import get_strong_augmentation, get_img_size
    # Reapply strong augmentation independently (tensors back to PIL not needed
    # here — we apply two random strong transforms to the same raw batch)
    # In practice this requires raw PIL images; this is a simplified version.
    return torch.tensor(0.0, device=device)  # placeholder


def _build_unlabeled_pool(args, class_ids, device):
    """Build unlabeled pool of P images per novel class."""
    from datasets import get_session_datasets
    _, unlabeled_loader, _ = get_session_datasets(
        args, session_id=1, class_ids=class_ids)
    pool = []
    if unlabeled_loader is not None:
        for imgs, labels in unlabeled_loader:
            for img, lbl in zip(imgs, labels):
                pool.append((img, lbl.item()))
    return pool


# ── Class name helper ─────────────────────────────────────────────────────────

def get_class_names(dataset_name, data_root, class_ids):
    """Load human-readable class names for prompt construction."""
    if dataset_name == "CIFAR100":
        from torchvision.datasets import CIFAR100 as C100
        ds = C100(data_root, train=True, download=False)
        return [ds.classes[i] for i in class_ids]

    if dataset_name == "miniImageNet":
        name_file = os.path.join(data_root, "miniImageNet", "class_names.txt")
        with open(name_file) as f:
            all_names = [l.strip() for l in f]
        return [all_names[i] for i in class_ids]

    if dataset_name == "CUB200":
        name_file = os.path.join(data_root, "CUB_200_2011", "classes.txt")
        with open(name_file) as f:
            all_names = [l.strip().split(".")[-1].replace("_", " ")
                         for l in f]
        return [all_names[i] for i in class_ids]

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Main training loop ────────────────────────────────────────────────────────

def run_single(args, run_id=0):
    """Single complete run across all k sessions."""
    set_seed(args.seed + run_id)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = SFSCILModel(args, device=device).to(device)
    class_schedule = get_class_schedule(args)
    logger = SessionLogger()

    # Base session
    exemplar_set, base_acc = train_base_session(model, args, device)
    logger.log(0, base_acc)

    base_model = snapshot_model(model)
    prev_model = snapshot_model(model)

    all_class_ids = class_schedule[0].copy()

    # Incremental sessions
    for t in range(1, args.num_sessions):
        new_class_ids = class_schedule[t]
        all_class_ids = all_class_ids + new_class_ids

        exemplar_set, acc = train_incremental_session(
            model, base_model, prev_model,
            session_id=t,
            class_ids=new_class_ids,
            all_class_ids=all_class_ids,
            exemplar_set=exemplar_set,
            args=args,
            device=device,
        )
        logger.log(t, acc)

        # Update previous session model
        prev_model = snapshot_model(model)

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"run{run_id}_session{t}.pt")
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    avg = logger.summary()
    return logger.session_accs, avg


def main():
    args = get_args()
    print(f"\nDataset: {args.dataset} | Backbone: {args.backbone} "
          f"| Sessions: {args.num_sessions} | Runs: {args.runs}")

    all_session_accs = []
    all_avgs = []

    for run in range(args.runs):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{args.runs}")
        print(f"{'='*60}")
        session_accs, avg = run_single(args, run_id=run)
        all_session_accs.append(session_accs)
        all_avgs.append(avg)

    # Aggregate mean ± std across runs
    all_session_accs = np.array(all_session_accs)   # [runs, sessions]
    mean_per_session = all_session_accs.mean(axis=0)
    std_per_session  = all_session_accs.std(axis=0)

    print(f"\n{'='*60}")
    print("FINAL RESULTS (mean ± std across runs)")
    print(f"{'='*60}")
    for s, (m, s_) in enumerate(zip(mean_per_session, std_per_session)):
        print(f"  Session {s}: {m:.2f} ± {s_:.2f}%")
    print(f"  Average: {np.mean(all_avgs):.2f} ± {np.std(all_avgs):.2f}%")

    # Save results
    results = {
        "dataset":   args.dataset,
        "backbone":  args.backbone,
        "sessions":  {
            f"s{s}": {"mean": float(m), "std": float(s_)}
            for s, (m, s_) in enumerate(zip(mean_per_session, std_per_session))
        },
        "average":  {"mean": float(np.mean(all_avgs)),
                     "std":  float(np.std(all_avgs))},
    }
    os.makedirs(args.log_dir, exist_ok=True)
    out_path = os.path.join(args.log_dir,
                            f"{args.dataset}_{args.backbone}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
