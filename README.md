**SFSCIL:Vision-Language Geometry as a Shared Anchor for Semi-Supervised Class-Incremental Learning via Knowledge Distillation**
This repository contains the official implementation of our SFSCIL framework. The codebase supports benchmarks such as CUB-200, CIFAR-100 and miniImageNet, implementing a Vision Transformer (ViT) backbone fused with CLIP embeddings, incremental classification, and distillation-based replay.

**Requirements**
Ensure you have Python 3.8+ installed. Install the required dependencies via:
pip install -r requirements.txt
run the training loop on miniImageNet:
python train.py --dataset miniImageNet
The training script (train.py) executes 4 independent runs by default (--runs 4) and reports the mean accuracy ± standard deviation.

**Datasets**
We follow FSCIL setting to use the same data index_list for training. Please follow the guidelines in CEC to prepare them. Scripts for experiments on mini-imagenet are as follows, and the full codes will be available upon acceptance:
📈 **Training Details**
The training process follows the Session-Incremental protocol:
Base Session: Train on base classes.
Incremental Sessions: New classes are introduced; old classes are preserved via exemplar replay and distillation.
Pseudo-Labeling: Unlabeled data is leveraged using the two-stage selection mechanism defined in utils.py.
🙏 **Acknowledgments**
Our project references the codes in the following repositories:
fscil
CEC
FACT
**Citation**
If you use this code in your research, please cite our paper:
@article{yourpaper2024,
  title={Vision-Language Geometry as a Shared Anchor for Semi-Supervised Class-Incremental Learning via Knowledge Distillation},
  author={Feidu Akmel, Xun Gong},
  Journal={Signal Processing},
  year={2026}
}
