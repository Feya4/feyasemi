"""
config.py — S-FSCIL: Vision-Language Geometry as a Shared Anchor
for Semi-Supervised Few-Shot Class-Incremental Learning
"""

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="S-FSCIL Training")

    # ── Dataset ───────────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="miniImageNet",
                        choices=["miniImageNet", "CIFAR100", "CUB200"],
                        help="Dataset name")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--num_workers", type=int, default=4)

    # ── FSCIL protocol ────────────────────────────────────────────────────────
    parser.add_argument("--base_classes", type=int, default=60,
                        help="Number of base classes (session 0)")
    parser.add_argument("--way", type=int, default=5,
                        help="N-way for incremental sessions")
    parser.add_argument("--shot", type=int, default=5,
                        help="K-shot for incremental sessions")
    parser.add_argument("--num_sessions", type=int, default=9,
                        help="Total sessions including base (k)")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--backbone", type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "resnet12"],
                        help="Feature extractor backbone")
    parser.add_argument("--freeze_layers", type=int, default=4,
                        help="Number of ViT layers to freeze in incremental sessions")
    parser.add_argument("--mlp_hidden", type=int, default=512,
                        help="Hidden dimension of logit fusion MLP")

    # ── CLIP ──────────────────────────────────────────────────────────────────
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP model variant (frozen throughout)")
    parser.add_argument("--tau_clip", type=float, default=0.01,
                        help="CLIP semantic similarity temperature τ")
    parser.add_argument("--prompt_ensemble", action="store_true", default=True,
                        help="Use prompt ensembling (K=7 templates) for CUB-200")

    # ── Semi-supervised ───────────────────────────────────────────────────────
    parser.add_argument("--unlabeled_pool", type=int, default=50,
                        help="Unlabeled pool size P per novel class")
    parser.add_argument("--selection_ratio", type=float, default=0.25,
                        help="Fraction S=0.25P of pool to select")
    parser.add_argument("--tau_c", type=float, default=0.95,
                        help="Confidence threshold τ_c for pseudo-label gating")
    parser.add_argument("--mu", type=int, default=2,
                        help="Unlabeled batch multiplier: B_u = μ * B_l")

    # ── Distillation ──────────────────────────────────────────────────────────
    parser.add_argument("--T", type=float, default=2.0,
                        help="Distillation temperature T")

    # ── Exemplar memory ───────────────────────────────────────────────────────
    parser.add_argument("--memory_size", type=int, default=1,
                        help="Exemplar budget M per class")

    # ── Loss weights ──────────────────────────────────────────────────────────
    parser.add_argument("--lambda_l", type=float, default=1.0,
                        help="Supervised loss weight")
    parser.add_argument("--lambda_u", type=float, default=1.0,
                        help="Unsupervised pseudo-label loss weight")
    parser.add_argument("--lambda_c", type=float, default=0.1,
                        help="Consistency regularization weight")
    parser.add_argument("--lambda_d", type=float, default=1.0,
                        help="Distillation loss weight")

    # ── Optimisation — base session ───────────────────────────────────────────
    parser.add_argument("--base_epochs", type=int, default=220,
                        help="Epochs for base session (60 for CUB-200)")
    parser.add_argument("--base_lr", type=float, default=0.1,
                        help="Initial learning rate for base session")
    parser.add_argument("--lr_decay_epochs", type=int, nargs="+",
                        default=[80, 120],
                        help="Epochs at which LR is divided by 10")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size_l", type=int, default=32,
                        help="Labeled batch size B_l")

    # ── Optimisation — incremental sessions ──────────────────────────────────
    parser.add_argument("--inc_epochs", type=int, default=5,
                        help="Epochs per incremental session")
    parser.add_argument("--inc_lr", type=float, default=0.001,
                        help="Learning rate for incremental sessions (0.0005 CUB-200)")

    # ── Augmentation ──────────────────────────────────────────────────────────
    parser.add_argument("--randaugment_n", type=int, default=2,
                        help="RandAugment N (strong augmentation)")
    parser.add_argument("--randaugment_m", type=int, default=10,
                        help="RandAugment M (strong augmentation)")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=4,
                        help="Number of independent runs for mean±std reporting")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Dataset-specific overrides
    if args.dataset == "CUB200":
        args.base_classes = 100
        args.way = 10
        args.num_sessions = 11
        args.base_epochs = 60
        args.base_lr = 0.001
        args.inc_lr = 0.0005

    if args.dataset == "CIFAR100":
        args.base_classes = 60

    return args


# Prompt templates (K=7) used for CUB-200 ensembling
CUB_PROMPT_TEMPLATES = [
    "a photo of a {}, a type of bird",
    "a close-up photo of a {}",
    "a wildlife photo of a {}",
    "a {}, a type of bird",
    "a photo of the {}, a bird species",
    "an image of a {}",
    "a photo of a {} bird",
]

# Standard single template for CIFAR-100 and miniImageNet
DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"
