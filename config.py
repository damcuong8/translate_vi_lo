from pathlib import Path
import math

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 20,
        "lr": 5e-4,
        "max_len": 500,
        
        # Tham số mô hình
        "d_model": 384,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.15,
        "d_ff": 1536,
        

        "weight_decay": 0.01,  # L2 regularization
        "label_smoothing": 0.1,  # Label smoothing
        "lr_scheduler": "cosine",
        "warmup_steps": 4000,
        "gradient_clip_val": 1.0,  # Gradient clipping
        
        # Regularization và data augmentation
        "word_dropout_rate": 0.1,  # Random word dropout
        
        # Checkpointing và evaluation
        "save_strategy": "steps",
        "save_steps": 1000,
        "evaluation_strategy": "steps",
        "eval_steps": 1000,
        
        # Mixed precision training
        "use_mixed_precision": True,  # Sử dụng mixed precision
        
        # Early stopping parameters
        "early_stopping": True,  # Whether to use early stopping
        "early_stopping_patience": 10,  # Number of epochs with no improvement to stop training
        "early_stopping_metric": "bleu",  # Metric to monitor: 'bleu', 'wer', 'cer', or 'loss'
        "early_stopping_min_delta": 0.0001,  # Minimum change to be considered as improvement
        "save_best_model": True,  # Whether to save the best model
        
        "datasource": 'vi_lo',  # Vietnamese-Lao
        "lang_src": "vi",       # Vietnamese
        "lang_tgt": "lo",       # Lao
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        
        # Paths to the tokenizer models
        "tokenizer_src_path": "kaggle/working/tokenizer_vi.model",  # Path to Vietnamese tokenizer
        "tokenizer_tgt_path": "kaggle/working/tokenizer_lo.model",  # Path to Lao tokenizer
        
        # Paths to data files
        "train_src_file": "VLSP2023/Train/train2023.vi",  # Vietnamese training data
        "train_tgt_file": "VLSP2023/Train/train2023.lo",  # Lao training data
        "val_src_file": "VLSP2023/Dev/dev2023.vi",      # Vietnamese validation data
        "val_tgt_file": "VLSP2023/Dev/dev2023.lo",      # Lao validation data
        
        "experiment_name": "runs/vi_lo_model"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
