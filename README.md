# ViT-torch: Vision Transformer on CIFAR-10 (PyTorch)

This project is a complete implementation of Vision Transformer (ViT) applied to small-scale datasets (especially CIFAR-10), including:

- Model implementations with various configurations (native ViT, ResNet+ViT hybrid, different patch/heads/blocks setups, Stochastic Depth/DropPath, etc.)
- Training and evaluation scripts (with learning rate schedulers: Warmup/Linear/Cosine/Constant-Cosine/Warmup-Constant-Cosine)
- Data augmentation (RandomCrop+Paste, MixUp, CutMix, RandAugment, and batch random augmentation)
- Visualization and analysis (attention maps, attention distance, gradient rollout, feature maps, positional embedding similarity)

## Contributors

- Junjie Yu (GitHub: [JunjieYu28](https://github.com/JunjieYu28)) - Primarily responsible for data augmentation and parameter tuning.
- Xunakun Yang (GitHub: [xuankunyang](https://github.com/xuankunyang)) - Primarily responsible for model architecture and variants, various training methods, and parameter tuning.

## Project Report and Presentation

- Report PDF: [ViT.pdf](https://github.com/xuankunyang/ViT-on-CIFAR-10/blob/main/ViT.pdf)

- Presentation PPT: [ViT.pptx](https://github.com/xuankunyang/ViT-on-CIFAR-10/blob/main/ViT.pptx)


## Directory Structure Overview

```
ViT_torch/
  left/                     # Early/basic training scripts and utilities
    data_utils.py           # CIFAR-10/100 data loading (basic augmentation)
    train.py                # Basic training entry (using models/modeling.py)
    train_aug.py            # Augmentation training based on basic pipeline
    train_aug_pro.py        # Advanced augmentation training
    train_sd.py             # Stochastic Depth experiments
    random_aug.py           # Additional RandAug definitions
  models/
    configs.py              # Model hyperparameter configurations (patch/hidden/heads/layers etc.)
    modeling.py             # Main ViT implementation (with optional ResNet hybrid features)
    modeling_sd.py          # Stochastic Depth version
    model_final.py          # Final ViT version (small image friendly, img_size=32)
  utils/
    data_aug.py             # Enhanced data loading (MixUp/CutMix/RandomCropPaste/RandAugment)
    aug_utils.py            # Implementations for MixUp/CutMix/RandomCropPaste
    scheduler.py            # Learning rate schedulers (Warmup/Cosine/Constant-Cosine etc.)
    augment_images_all/     # Augmentation example images
  scripts/                  # Training script examples (.sh)
  train_final.py            # Enhanced training entry (defaults to model_final.py and utils/data_aug.py)
  compute_attention_distance_for_all.py  # Attention distance analysis
  grad_rollout.py           # Gradient Rollout visualization
  Visualization notebooks:
    visualize_attention_map.ipynb
    visualize_attention_distance.ipynb
    visualize_embedding_filters.ipynb
    visualize_feature_map.ipynb
    visualize_grad_rollout.ipynb
  ViT.pdf / ViT.pptx        # Report and presentation documents
```

## Environment and Dependencies

- Python ≥ 3.8
- PyTorch, Torchvision (CUDA recommended)
- Others: numpy, ml-collections, scipy, tqdm, tensorboard, scikit-learn, seaborn, matplotlib

Example installation:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # Adjust based on your CUDA version
pip install numpy ml-collections scipy tqdm tensorboard scikit-learn seaborn matplotlib
```

## Data

No manual download needed; CIFAR-10/100 will be automatically downloaded to ./data on first run.

## Quick Start (Recommended: Final Pipeline)

`train_final.py` uses the small-image-friendly implementation from `models/model_final.py` and augmented data pipeline from `utils/data_aug.py`, default img_size=32, with richer visualization stats and augmentation strategies.

Minimal example (CIFAR-10):

```
python train_final.py --name exp_final_c10 \
  --dataset cifar10 \
  --model_type ViT-Ours_Final \
  --img_size 32 \
  --train_batch_size 512 --eval_batch_size 1024 \
  --learning_rate 1e-3 --weight_decay 5e-5 \
  --num_steps 20000 --decay_type cosine --warmup_steps 500 \
  --aug_type batch_random --random_aug true
```

Key parameters (excerpt):

- `--model_type`: See "Models and Configurations" below
- `--decay_type`: cosine | linear | constant_cosine | warmup_constant_cosine
- `--aug_type`: None | mixup | random_crop_paste | cutmix | batch_random
- `--random_aug`: Enable RandAugment (can coexist with above augmentations)
- `--mixup_rate`, `--cutmix_rate`, `--cut_rate`, `--flip_p`: Hyperparams for MixUp, CutMix, RandomCropPaste

Outputs:

- Model weights: output_final/{name}_checkpoint.bin
- TensorBoard logs: logs/{name} (includes Top-1/Top-5, confusion matrices, config snapshots, etc.)

View training curves:

```
tensorboard --logdir logs
```

## Basic Pipeline (For Comparison)

`left/train.py` uses `models/modeling.py` and `left/data_utils.py` (more basic augmentations), suitable for baseline experiments:

```
python left/train.py --name exp_base_c10 \
  --dataset cifar10 \
  --model_type ViT-B_16 \
  --img_size 32 \
  --train_batch_size 512 --eval_batch_size 64 \
  --learning_rate 3e-2 --weight_decay 0 \
  --num_steps 10000 --decay_type warmup_constant_cosine --warmup_steps 500
```

Outputs: output/{name}_checkpoint.bin and logs/{name}.

## Models and Configurations (Excerpt)

Select configurations via CONFIGS dict in `models/model_final.py` and `models/modeling.py`:

Final versions and variants (from model_final.py → configs.py):

- ViT-Ours_Final, ViT-Ours_sd{0..4}, ViT-Ours_dp{0..3}, ViT-Ours_adp{0..3}, ViT-Ours_res{0..2}
- ViT-Ours_ps{2,4,8} (different patch sizes), ViT-Ours_nb{4,12} (different layers), ViT-Ours_nh{8,16} (different heads)
- ViT-Ours_set_288_288/384/768, ViT-Ours_set_384_768 etc.

Basic versions (from modeling.py → configs.py):

- ViT-Ours_Res, ViT-Ours, ViT-Ours_new, ViT-B_16/B_32, ViT-L_16/L_32, ViT-H_14, R50-ViT-B_16

Notes:

- If config has ResNet_type != 0 or patches.grid, uses ResNet features as hybrid input (Hybrid ViT).
- transformer.prob_pass > 0 enables random layer skipping during training (approx. DropPath / Stochastic Depth).

## Data Augmentation Details

`utils/aug_utils.py`:

- RandomCropPaste(size, alpha, flip_p): Random crop-flip-paste within same image with local mixing; good for "structural perturbation" on small images.
- MixUp(alpha), CutMix(beta): Standard mixing augmentations; loss computed automatically in batch logic in train_final.py.

`utils/data_aug.py`:

- `--aug_type` switches strategies; batch_random randomly selects from random_crop_paste, mixup, cutmix.
- `--random_aug true` enables RandAugment, stackable with above.

Example visualizations: See utils/augment_images_all/ for sample images.

## Learning Rate Scheduling

Provided by `utils/scheduler.py`:

- WarmupLinearSchedule, WarmupCosineSchedule(min_lr)
- ConstantCosineSchedule(constant_steps, min_lr)
- WarmupConstantCosineSchedule(warmup_steps, constant_steps, min_lr)

Enabled via `--decay_type` with `--warmup_steps`, `--constant_steps`, `--min_lr`.

## Evaluation and Visualization

Evaluates on validation set every `--eval_every` steps; train_final.py logs Top-1/Top-5, confusion matrices, and saves best weights.

Attention visualization and analysis:

- Jupyter Notebooks: visualize_attention_map.ipynb, visualize_grad_rollout.ipynb, etc.
- compute_attention_distance_for_all.py: Attention distance analysis. Note: checkpoint_path has example hardcoded paths; modify to your local model paths.

## Multi-GPU/Distributed

Scripts support `--local_rank` for distributed; keep `--local_rank -1` (default) for non-distributed.

## Common Issues (FAQ)

- Out of memory/GPU memory: Reduce `--train_batch_size` or increase `--gradient_accumulation_steps`; choose smaller model configs (fewer layers/lower hidden).
- Training not converging: Adjust `--learning_rate`, `--weight_decay`, `--warmup_steps`; disable strong augmentations first (e.g., `--aug_type None` for baseline).
- Running .sh on Windows: .sh are example command sets; on Windows, run python commands directly with parameters.

## References and Acknowledgments

Co-authors: Xuankun Yang (GitHub: [xuankunyang](https://github.com/xuankunyang)) and Junjie Yu (GitHub: [JunjieYu28](https://github.com/JunjieYu28)). Thanks for collaborative work on model implementation, training, and visualization.


ViT original ideas from Google's Vision Transformer paper and open-source implementation (some configs and naming styles follow theirs).

## License

This project includes significant modifications and extensions based on open-source implementations, released under the MIT License. See LICENSE file in the repository root.

Original upstream project license is MIT; retained in LICENSE.

Copyright for modifications and new code: Copyright (c) 2025 Junjie Yu and Xuankun Yang