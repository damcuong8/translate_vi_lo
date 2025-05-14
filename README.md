# Vietnamese-Lao Neural Machine Translation

Transformer-based machine translation system for Vietnamese-Lao optimized for ~100K sentence pairs.

## Model

- 4-layer Transformer (encoder-decoder)
- 384d embeddings, 8 heads, 1536d FFN
- SentencePiece tokenization
- Dynamic batching with on-the-fly padding
- Mixed precision training & cosine LR schedule


## Usage

Training:
```bash
python train.py
```

Translation:
```bash
python translate.py --source input.vi --output output.lo --model vi_lo_weights/tmodel_best.pt
# Interactive mode
python translate.py --interactive
```

## Key Configs

```python
{
    "batch_size": 16,
    "num_epochs": 20,
    "lr": 5e-4,
    "d_model": 384,
    "num_layers": 4,
    "dropout": 0.15,
    "warmup_steps": 4000
}
```

## Features

- Word dropout regularization
- Early stopping based on BLEU score
- Checkpointing system (epoch/step/best)
- Evaluation metrics: BLEU, WER, CER

