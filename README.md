# Vietnamese-Lao Neural Machine Translation

A Transformer-based neural machine translation system for Vietnamese-Lao language pair, optimized for datasets with ~100K sentence pairs. This project implements a full sequence-to-sequence Transformer architecture with dynamic batching, mixed precision training, and beam search decoding.

## Model Architecture

- **Transformer:** 4-layer encoder-decoder architecture
- **Dimensions:** 384d model size, 8 attention heads, 1536d feed-forward network
- **Regularization:** Dropout (0.15), word dropout, label smoothing (0.1)
- **Tokenization:** SentencePiece subword tokenization 
- **Optimizer:** AdamW with weight decay (0.01) and gradient clipping (1.0)
- **Learning rate:** Cosine schedule with linear warmup (4000 steps)

## Features

- **Dynamic batching** with on-the-fly padding for efficient training
- **Mixed precision training** for faster computation
- **Distributed training** with multi-GPU support using DistributedDataParallel
- **Gradient accumulation** for effective larger batch sizes
- **Early stopping** based on BLEU, WER, or CER metrics
- **Comprehensive evaluation** with multiple metrics
- **Beam search decoding** for better translation quality
- **Checkpointing system** (epoch/step/best model saving)


## Usage

### Training

```bash
# Basic training
python train.py

# Advanced training options
python train.py --batch_size 32 --epochs 30 --distributed
```

### Translation

```bash
# Translate a single sentence with beam search
python translate.py --text "Tôi yêu đất nước Việt Nam" --beam 5

# Translate a file
python translate.py --source input.vi --output output.lo --model vi_lo_weights/tmodel_best.pt --beam 3

# Interactive mode
python translate.py --interactive --beam 5
```

### Evaluation

Evaluate on test set with BLEU, WER, and CER metrics:

```bash
python evaluate.py --test_file path/to/test.vi --reference path/to/reference.lo
```

## Configuration

Key configuration parameters in `config.py`:

```python
{
    # Training parameters
    "batch_size": 16,         # Per GPU batch size
    "num_epochs": 20,
    "lr": 5e-4,
    "warmup_steps": 4000,
    "gradient_clip_val": 1.0,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    
    # Model architecture
    "d_model": 384,           # Model dimension
    "num_layers": 4,          # Number of encoder/decoder layers
    "num_heads": 8,           # Number of attention heads
    "dropout": 0.15,
    "d_ff": 1536,             # Feed-forward dimension
    
    # Advanced features
    "distributed_training": True,
    "word_dropout_rate": 0.1,
    "use_mixed_precision": True,
    
    # Early stopping
    "early_stopping": True,
    "early_stopping_metric": "bleu",
    "early_stopping_patience": 10,
}
```

## Beam Search

The implementation includes beam search decoding to improve translation quality. Key features:

- Maintains multiple translation candidates at each step
- Length normalization to avoid bias toward shorter sequences
- Configurable beam size (default: 5)

Example usage:

```bash
# Use beam search with size 5
python translate.py --text "Hôm nay trời đẹp" --beam 5
```

## Performance

On a dataset of ~100K Vietnamese-Lao sentence pairs:

| Metric | Score |
|--------|-------|
| BLEU   | 28.7  |
| WER    | 0.44  |
| CER    | 0.32  |

Training time: ~3 hours on a single NVIDIA T4 GPU.

## Distributed Training

For multi-GPU training, the system automatically detects available GPUs and distributes the workload:

```bash
# Enable distributed training on all available GPUs
python train.py

# Specify number of GPUs to use
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)
- Optimized for Vietnamese-Lao translation as part of the VLSP 2023 shared task

