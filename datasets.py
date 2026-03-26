"""
datasets.py — Dataset classes for S-FSCIL
Supports: miniImageNet, CIFAR-100, CUB-200
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from timm.data.auto_augment import rand_augment_transform


# ── Augmentation builders ─────────────────────────────────────────────────────

def get_weak_augmentation(dataset_name, img_size=84):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(*get_mean_std(dataset_name)),
    ])


def get_strong_augmentation(dataset_name, img_size=84, n=2, m=10):
    ra = rand_augment_transform(
        config_str=f"rand-n{n}-m{m}-mstd0.5",
        hparams={"translate_const": int(img_size * 0.45)},
    )
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=8),
        ra,
        transforms.ToTensor(),
        transforms.Normalize(*get_mean_std(dataset_name)),
    ])


def get_test_transform(dataset_name, img_size=84):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*get_mean_std(dataset_name)),
    ])


def get_mean_std(dataset_name):
    stats = {
        "miniImageNet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "CIFAR100":     ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        "CUB200":       ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    }
    return stats[dataset_name]


def get_img_size(dataset_name):
    return {"miniImageNet": 84, "CIFAR100": 32, "CUB200": 224}[dataset_name]


# ── Base dataset class ────────────────────────────────────────────────────────

class FewShotDataset(Dataset):
    """Generic few-shot dataset wrapper."""

    def __init__(self, data, labels, transform=None):
        self.data = data        # list of PIL images or file paths
        self.labels = labels    # list of integer labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── miniImageNet ──────────────────────────────────────────────────────────────

class MiniImageNet(Dataset):
    """
    miniImageNet: 100 classes, 600 images/class, 84×84.
    Split: base 60 classes, 8 incremental sessions of 5-way 5-shot.
    Expects data organized as:
        data_root/miniImageNet/images/<classname>/<img>.jpg
        data_root/miniImageNet/split/base.txt  (class names, one per line)
        data_root/miniImageNet/split/train.csv / test.csv
    """

    def __init__(self, data_root, split="train", transform=None,
                 class_ids=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []

        split_file = os.path.join(data_root, "miniImageNet", "split",
                                  f"{split}.csv")
        img_dir = os.path.join(data_root, "miniImageNet", "images")

        class_to_idx = {}
        with open(split_file) as f:
            next(f)  # skip header
            for line in f:
                fname, cls = line.strip().split(",")
                if cls not in class_to_idx:
                    class_to_idx[cls] = len(class_to_idx)
                label = class_to_idx[cls]
                if class_ids is None or label in class_ids:
                    self.img_paths.append(os.path.join(img_dir, fname))
                    self.labels.append(label)

        self.class_to_idx = class_to_idx
        self.class_names = list(class_to_idx.keys())

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── CIFAR-100 ─────────────────────────────────────────────────────────────────

class CIFAR100FSCIL(Dataset):
    """
    CIFAR-100 adapted for FSCIL.
    Base: 60 classes. 8 incremental sessions of 5-way 5-shot.
    """

    def __init__(self, data_root, train=True, transform=None, class_ids=None):
        self.transform = transform
        base = CIFAR100(root=data_root, train=train, download=True)
        self.class_names = base.classes

        self.data = []
        self.labels = []
        for img, label in zip(base.data, base.targets):
            if class_ids is None or label in class_ids:
                self.data.append(img)   # numpy HWC
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── CUB-200 ───────────────────────────────────────────────────────────────────

class CUB200(Dataset):
    """
    CUB-200-2011: 200 bird species, 224×224.
    Base: 100 classes. 10 incremental sessions of 10-way 5-shot.
    Expects:
        data_root/CUB_200_2011/images/<class_folder>/<img>.jpg
        data_root/CUB_200_2011/images.txt
        data_root/CUB_200_2011/image_class_labels.txt
        data_root/CUB_200_2011/train_test_split.txt
    """

    def __init__(self, data_root, train=True, transform=None, class_ids=None):
        self.transform = transform
        cub_root = os.path.join(data_root, "CUB_200_2011")

        # Load file paths
        id2path = {}
        with open(os.path.join(cub_root, "images.txt")) as f:
            for line in f:
                img_id, path = line.strip().split()
                id2path[img_id] = os.path.join(cub_root, "images", path)

        # Load labels (1-indexed → 0-indexed)
        id2label = {}
        with open(os.path.join(cub_root, "image_class_labels.txt")) as f:
            for line in f:
                img_id, label = line.strip().split()
                id2label[img_id] = int(label) - 1

        # Load train/test split
        id2split = {}
        with open(os.path.join(cub_root, "train_test_split.txt")) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                id2split[img_id] = int(is_train)

        self.img_paths = []
        self.labels = []
        for img_id, path in id2path.items():
            label = id2label[img_id]
            split_flag = id2split[img_id]
            if (train and split_flag == 1) or (not train and split_flag == 0):
                if class_ids is None or label in class_ids:
                    self.img_paths.append(path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── Semi-supervised dataset wrapper ──────────────────────────────────────────

class SemiSupervisedDataset(Dataset):
    """
    Wraps labeled and unlabeled subsets for joint mini-batch construction.
    Returns (x_weak, x_strong, label, is_labeled) per sample.
    Unlabeled samples have label=-1.
    """

    def __init__(self, labeled_dataset, unlabeled_dataset,
                 weak_transform, strong_transform):
        self.labeled = labeled_dataset
        self.unlabeled = unlabeled_dataset
        self.weak_tf = weak_transform
        self.strong_tf = strong_transform

    def __len__(self):
        return len(self.labeled) + len(self.unlabeled)

    def __getitem__(self, idx):
        if idx < len(self.labeled):
            img, label = self.labeled[idx]
            # For labeled samples: weak view only (strong not used in L_l)
            return self.weak_tf(img), self.weak_tf(img), label, True
        else:
            img, _ = self.unlabeled[idx - len(self.labeled)]
            return self.weak_tf(img), self.strong_tf(img), -1, False


# ── Session data builder ──────────────────────────────────────────────────────

def get_session_datasets(args, session_id, class_ids, exemplar_set=None):
    """
    Build labeled, unlabeled, and test datasets for a given session.

    Args:
        args:        parsed config
        session_id:  int, 0 = base session
        class_ids:   list of class indices for this session
        exemplar_set: list of (img_tensor, label) from previous sessions

    Returns:
        labeled_loader, unlabeled_loader, test_loader
    """
    img_size = get_img_size(args.dataset)
    weak_tf   = get_weak_augmentation(args.dataset, img_size)
    strong_tf = get_strong_augmentation(args.dataset, img_size,
                                        args.randaugment_n, args.randaugment_m)
    test_tf   = get_test_transform(args.dataset, img_size)

    DatasetClass = {
        "miniImageNet": MiniImageNet,
        "CIFAR100":     CIFAR100FSCIL,
        "CUB200":       CUB200,
    }[args.dataset]

    # Labeled set for this session
    labeled_ds = DatasetClass(args.data_root, train=True,
                              transform=weak_tf, class_ids=set(class_ids))

    # Unlabeled pool for incremental sessions
    unlabeled_loader = None
    if session_id > 0:
        unlabeled_ds = DatasetClass(args.data_root, train=True,
                                    transform=None, class_ids=set(class_ids))
        # Each class: sample P images
        pool = _sample_pool(unlabeled_ds, class_ids, args.unlabeled_pool)
        pool_ds = FewShotDataset(
            [img for img, _ in pool],
            [lbl for _, lbl in pool],
            transform=None,   # transforms applied per-view in SemiSupervisedDataset
        )
        unlabeled_loader = DataLoader(pool_ds,
                                      batch_size=args.batch_size_l * args.mu,
                                      shuffle=True,
                                      num_workers=args.num_workers)

    # Add exemplars to labeled set
    if exemplar_set:
        ex_imgs, ex_labels = zip(*exemplar_set)
        exemplar_ds = FewShotDataset(list(ex_imgs), list(ex_labels),
                                     transform=weak_tf)
        from torch.utils.data import ConcatDataset
        labeled_ds = ConcatDataset([labeled_ds, exemplar_ds])

    labeled_loader = DataLoader(labeled_ds,
                                batch_size=args.batch_size_l,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=True)

    # Test set: all seen classes
    all_seen = list(range(args.base_classes))
    for s in range(1, session_id + 1):
        all_seen += list(range(
            args.base_classes + (s - 1) * args.way,
            args.base_classes + s * args.way,
        ))
    test_ds = DatasetClass(args.data_root, train=False,
                           transform=test_tf, class_ids=set(all_seen))
    test_loader = DataLoader(test_ds,
                             batch_size=64,
                             shuffle=False,
                             num_workers=args.num_workers)

    return labeled_loader, unlabeled_loader, test_loader


def _sample_pool(dataset, class_ids, pool_size):
    """Sample up to pool_size images per class from dataset."""
    class_buckets = {c: [] for c in class_ids}
    for idx in range(len(dataset)):
        img, lbl = dataset[idx]
        if lbl in class_buckets:
            class_buckets[lbl].append((img, lbl))
    pool = []
    for c, samples in class_buckets.items():
        pool.extend(random.sample(samples, min(pool_size, len(samples))))
    return pool
