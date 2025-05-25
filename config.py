from pathlib import Path
import math
import torch

def get_config():
    return {
        "batch_size": 16,  # Batch size per GPU - IMPORTANT: This is the TOTAL batch size, will be split across GPUs
                          # With 2 GPUs: each GPU gets batch_size/2 = 8 samples
                          # Minimum recommended: 4 (so each GPU gets at least 2 samples)
        "num_epochs": 100,
        "lr": 5e-4,
        "max_len": 500,
        
        # Tham số mô hình
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.1,
        "d_ff": 2048,
        

        "weight_decay": 0.0001,  # L2 regularization
        "label_smoothing": 0.1,  # Label smoothing
        "lr_scheduler": "cosine",
        "warmup_steps": 4000,
        "gradient_clip_val": 2.0,  # Gradient clipping
        
        # Multi-GPU training parameters
        "training_mode": "auto",  # 'auto', 'single', 'dataparallel', 'distributed'
                                  # - 'auto': Use DataParallel on Kaggle, DistributedDataParallel elsewhere
                                  # - 'single': Force single GPU/CPU training
                                  # - 'dataparallel': Force DataParallel (recommended for Kaggle)
                                  # - 'distributed': Force DistributedDataParallel (recommended for local/cluster)
        
        "distributed_training": False,  # Bật distributed training (legacy, use training_mode instead)
        "force_single_gpu": False,  # Set to True to force single GPU training (legacy, use training_mode='single' instead)
        "num_gpus": 2,             # Number of GPUs to use (will use min of this and available)
        "backend": "nccl",         # Backend for distributed training
        "nccl_timeout": 3600000,   # 1 hour timeout for NCCL operations (in ms)
        
        # Regularization và data augmentation
        "word_dropout_rate": 0.1,  # Random word dropout
        
        # Checkpointing và evaluation
        "save_strategy": "epoch",
        "save_steps": 1000,
        "evaluation_strategy": "epoch",
        "eval_steps": 1000,
        "save_only_best": True,  # Only save best model, not epoch models
        "keep_checkpoint_max": 1,  # Maximum number of checkpoints to keep (applies to epoch models if enabled)
        
        # Mixed precision training
        "use_mixed_precision": True,  # Sử dụng mixed precision
        
        # Early stopping parameters
        "early_stopping": True,  # Whether to use early stopping
        "early_stopping_patience": 10,  # Number of epochs with no improvement to stop training
        "early_stopping_metric": "bleu",  # Metric to monitor: 'bleu', 'wer', 'cer', or 'loss'
        "early_stopping_min_delta": 0.001,  # Minimum change to be considered as improvement
        "save_best_model": True,  # Whether to save the best model
        
        "datasource": "VLSP2023",
        "lang_src": "vi",       # Vietnamese
        "lang_tgt": "lo",       # Lao
        "model_folder": "vi_lo_weights",
        "model_basename": "tmodel_",
        
        # To load a model, set this to the full path of the .pt file
        # Example: "./VLSP2023_vi_lo_weights/tmodel_best.pt"
        # Set to None to start from scratch
        "preload": "/kaggle/input/MNT_vi_lo/tmodel_best.pt",
        
        # Paths to the tokenizer models
        "tokenizer_src_path": "./vi_lo_weights/tokenizer_vi.model",  # Path to Vietnamese tokenizer
        "tokenizer_tgt_path": "./vi_lo_weights/tokenizer_lo.model",  # Path to Lao tokenizer
        
        # Paths to data files
        "train_src_file": "./VLSP2023/Train/vi-lo.train.vi",  # Vietnamese training data
        "train_tgt_file": "./VLSP2023/Train/vi-lo.train.lo",  # Lao training data
        "val_src_file": "./VLSP2023/Dev/vi-lo.dev.vi",      # Vietnamese validation data
        "val_tgt_file": "./VLSP2023/Dev/vi-lo.dev.lo",      # Lao validation data
        
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

def get_kaggle_config():
    """Configuration optimized for Kaggle DataParallel training"""
    config = get_config()
    config.update({
        # Optimized for Kaggle's 2-GPU setup
        "batch_size": 8,           # Total batch size (4 per GPU)
        "gradient_accumulation_steps": 2,  # Effective batch = 8 * 2 = 16
        "training_mode": "dataparallel",   # Force DataParallel
        "num_gpus": 2,
        
        # Smaller model for Kaggle memory constraints
        "d_model": 384,
        "num_layers": 4,
        "d_ff": 1536,
        "max_len": 350,
        
        # Faster iteration
        "num_epochs": 50,
        "early_stopping_patience": 5,
        "warmup_steps": 2000,
        
        # Kaggle paths (update these for your dataset)
        "tokenizer_src_path": "/kaggle/input/your-dataset/tokenizer_vi.model",
        "tokenizer_tgt_path": "/kaggle/input/your-dataset/tokenizer_lo.model",
        "train_src_file": "/kaggle/input/your-dataset/vi-lo.train.vi",
        "train_tgt_file": "/kaggle/input/your-dataset/vi-lo.train.lo",
        "val_src_file": "/kaggle/input/your-dataset/vi-lo.dev.vi",
        "val_tgt_file": "/kaggle/input/your-dataset/vi-lo.dev.lo",
        "experiment_name": "runs/vi_lo_model_kaggle"
    })
    return config
