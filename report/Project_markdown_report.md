# Project 1 - ResNet-18 (CIFAR-10)

## Introduction
- Implement ResNet-18 from primitives (no torchvision.models) adapted for CIFAR-10 (32×32).
- Provide training loop, augmentation, checkpointing, and visualization: curves, confusion matrix, prediction grids, Grad-CAM heatmaps.
- Target: ≥80% test accuracy (needs ~100+ epochs or stronger augmentation + optimizer tuning and GPU).

## Code Details
- Implements ResNet-18 from scratch (BasicBlock) adapted for CIFAR-10 with 3×3 conv input.
- Loads CIFAR-10 with standard augmentation and normalization, providing train/test loaders.
- Uses SGD + LR scheduler + mixed-precision training, saving best and latest checkpoints.
- Generates visualizations: loss/accuracy curves, confusion matrix, prediction grids, Grad-CAM overlays.
- Full train/validation loop with checkpointing and artifact saving to runs/cls/.

## Output
- Files saved to runs/cls/:
    - latest.pth (last checkpoint)
    - best_model.pth (best val acc)
    - curves_cls.png (train/val loss + accuracy)
    - confusion_matrix.png
    - preds_grid.png
    - gradcam_0.png, gradcam_1.png ...
- Console prints of epoch metrics. On a single good GPU and using --epochs 100, you can expect to approach 80%+ if hyperparams and training are tuned (augmentations, lr schedule, weight decay, batch-size).

## Explanation & Practical Notes
- Architecture:
    - For CIFAR, initial 7×7 conv and maxpool from ImageNet ResNet are removed. Use 3×3 conv and small strides.
    - BasicBlock uses two 3×3 conv layers with identity/projection (1×1 conv) when downsampling (stride=2).
    - Final AdaptiveAvgPool2d((1,1)) ensures spatial dims collapse to 1×1.
- Training:
    - Use SGD + momentum + weight decay. LR schedule with MultiStepLR at epochs 30 and 45 is standard baseline.
    - To reach ≥80%: train 80–200 epochs, use stronger augmentation (Cutout, AutoAugment), or swap to cosine annealing and use Label Smoothing. Also, larger batch sizes + warmup can help.
- Grad-CAM:
    - The provided Grad-CAM hooks a high-level block (layer4) and averages gradients spatially.
    - For higher-fidelity maps, use deeper feature layers or upsample with smoother interpolation.
- Confusion Matrix:
    - Normalized by true rows to show per-class recall; diagonal dominance indicates good classification.
- GPU:
    - Mandatory for reasonable training time. On CPU, training is extremely slow and accuracy targets won't be practical.

## Sources Consulted
- He et al., Deep Residual Learning for Image Recognition (ResNet). https://arxiv.org/abs/1512.03385
- PyTorch docs – layers & best practices. https://pytorch.org/docs/stable/nn.html
- CIFAR-10 loading tutorial (user-requested). https://www.geeksforgeeks.org/python/how-to-load-cifar10-dataset-in-pytorch/
- Grad-CAM paper. https://arxiv.org/abs/1610.02391
- Typical CIFAR preprocessing stats (mean/std): multiple sources (common values used in practice).

## Key Learnings/Insights
- Removing the ImageNet stem (7×7 conv + pool) is necessary for small images.
- Identity vs projection shortcuts matter when changing channels/stride.
- Global average pooling makes model robust to final feature map size.
- Reaching 80% requires substantial training time or stronger augmentation.

## Conclusion
- Provided code implements ResNet-18 from primitives, training loop, and visualization. To meet acceptance criteria you must run the training on GPU for many epochs and optionally add augmentation improvements and stronger schedulers.

---

# Project 2 - Transformer Toy MT

## Introduction
- Implement encoder-decoder Transformer from scratch using torch.nn.Linear, LayerNorm, etc. (no nn.Transformer).
- Train on a small toy parallel corpus (I include a small synthetic dataset generator and a small English↔Pseudo-French set example).
- Save loss curves, attention heatmaps, masks demo, decode comparison table, and compute corpus BLEU (simple implementation).
- Goal: BLEU ≥ 15 on toy dataset (needs dataset selection; I provide code and pipeline; you may need to enlarge corpus to reach BLEU target).

## Code Details
- Minimal Transformer encoder-decoder built from scratch with PyTorch primitives.
- Synthetic parallel dataset generator with dynamic vocabulary and special tokens.
- Training/evaluation pipeline with cross-entropy loss, Adam optimizer, and BLEU scoring.
- Generates visual artifacts: attention maps, loss curves, decoded samples, and masks.

## Output
- Files under runs/mt/:
    - latest.pth (model)
    - curves_mt.png (train/val loss)
    - attention_layer0_head0.png (example cross-attention heatmap)
    - masks_demo.png
    - decodes_table.png
    - bleu_report.png
- Console prints including epoch BLEU and loss.

## Explanation & Practical Notes
- Model:
    - Uses learned token embeddings + sinusoidal positional encodings (classic).
    - MultiHeadAttention implemented manually: linear projections, split into heads, scaled dot-product, softmax, combine heads.
    - Decoder has both masked self-attention and cross-attention to encoder memory.
- Dataset:
    - I provided a synthetic toy dataset generator (reverse-word translation). For meaningful BLEU, you may want to use small real parallel corpora (e.g., Tatoeba, or a small subset of Multi30k).
- Masking:
    - Padding masks prevent attending to pad tokens; look-ahead masks enforce causality in decoder self-attention.
- Training:
    - Use Adam, lr 1e-3. For larger datasets increase d_model, layers, regularize, longer training.
- BLEU:
    - Provided simple corpus BLEU (unigram..4-gram). For production, use sacrebleu or nltk.translate.bleu_score.corpus_bleu for more robust measures.
- Attention visualization:
    - Visualize cross-attention (decoder queries vs encoder keys) to inspect alignment patterns; good toy tasks often show clear diagonal bands when source and target align.

## Sources Consulted
- Vaswani et al., Attention Is All You Need. https://arxiv.org/abs/1706.03762
- PyTorch tutorials (Transformer / seq2seq). https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- BLEU original paper (for concept): https://aclanthology.org/P02-1040/
- Various blog posts and notes on implementing attention from scratch (for reference).

## Key Learnings/Insights
- Attention maps provide interpretable alignment; cross-attention often shows clear bands if model learned alignment.
- Implementing masks correctly (padding × lookahead) is error-prone; visual mask debugging helps.
- Toy datasets can validate architecture & training pipeline but achieving high BLEU needs richer datasets & careful tokenization.

## Conclusion
- The script implements a minimal working Transformer that trains on toy data and saves required visual artifacts. To meet BLEU ≥15, use a larger/more realistic small corpus and train longer, or adjust model capacity & hyperparameters.

