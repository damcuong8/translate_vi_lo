from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
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
