from model import build_transformer
from dataset import BilingualDataset, causal_mask, collate_batch
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import functools
import math
import time

# Distributed training imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data.distributed import DistributedSampler

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# SentencePiece for tokenization
import sentencepiece as spm

# Import metrics from torchmetrics.text instead of torchmetrics directly
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Performs greedy decoding on the input source sentence.
    
    Args:
        model: The model to use for decoding (can be standard model, DDP wrapped model, or DP wrapped model)
        source: The source tensor
        source_mask: The source mask tensor
        tokenizer_src: The source tokenizer
        tokenizer_tgt: The target tokenizer
        max_len: The maximum length of the generated sentence
        device: The device to use
        
    Returns:
        The decoded output tensor
    """
    # Check if the model is wrapped in DDP or DP
    if hasattr(model, 'module'):
        # Use model.module for DDP and DP models
        return _greedy_decode_internal(model.module, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device)
    else:
        # Use model directly for standard models
        return _greedy_decode_internal(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device)

def _greedy_decode_internal(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Internal implementation of greedy decoding to be used by the wrapper function
    """
    sos_idx = tokenizer_tgt.piece_to_id("<s>")
    eos_idx = tokenizer_tgt.piece_to_id("</s>")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=16, num_display=5):
    """
    Runs validation on the validation dataset
    
    Args:
        model: The model to validate
        validation_ds: The validation dataset
        tokenizer_src: The source tokenizer
        tokenizer_tgt: The target tokenizer
        max_len: The maximum length for generated sequences
        device: The device to run validation on
        print_msg: A function to print messages
        global_step: The current global step
        writer: The tensorboard writer
        num_examples: The number of examples to evaluate
        num_display: The number of examples to display
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    
    # For proper BLEU calculation
    tokenized_expected = []
    tokenized_predicted = []

    print_msg("-" * 80)
    print_msg(f"Running validation on {num_examples} examples, displaying {num_display} examples")
    
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy().tolist())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Tokenize for BLEU score calculation (ensure these are string inputs)
            # Simple tokenization by splitting on spaces for each text string
            if isinstance(target_text, str):
                tokenized_expected.append(target_text.split())
            else:
                tokenized_expected.append([str(target_text)])  # Fallback
                
            if isinstance(model_out_text, str):
                tokenized_predicted.append(model_out_text.split())
            else:
                tokenized_predicted.append([str(model_out_text)])  # Fallback
            
            # Print only the first few examples
            if count <= min(num_display, num_examples):
                print_msg('-'*console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                if num_display < num_examples:
                    print_msg('-'*console_width)
                    print_msg(f"... {count - num_display} more examples evaluated but not displayed ...")
                print_msg('-'*console_width)
                break
    
    # Initialize metrics
    metrics = {}
    
    # Always calculate metrics regardless of writer
    # Compute the char error rate 
    metric = CharErrorRate()
    cer = metric(predicted, expected)
    print_msg(f"Character Error Rate: {cer.item():.6f}")
    metrics['cer'] = cer.item()
    
    # Log to TensorBoard if writer exists
    if writer is not None:
        log_to_tensorboard(writer, 'validation/cer', cer, global_step)

    # Compute the word error rate
    metric = WordErrorRate()
    wer = metric(predicted, expected)
    print_msg(f"Word Error Rate: {wer.item():.6f}")
    metrics['wer'] = wer.item()
    
    # Log to TensorBoard if writer exists
    if writer is not None:
        log_to_tensorboard(writer, 'validation/wer', wer, global_step)

    # Compute the BLEU metric with properly tokenized input
    try:
        # Make sure inputs are properly formatted
        # BLEU expects references to be a list of lists
        references = [[t] for t in tokenized_expected]  # Wrap each reference in a list
        
        # Print debug info
        print_msg(f"Computing BLEU score with {len(tokenized_predicted)} hypotheses and {len(references)} references")
        if len(tokenized_predicted) > 0 and len(references) > 0:
            print_msg(f"Example hypothesis: {tokenized_predicted[0][:10]}")
            print_msg(f"Example reference: {references[0][0][:10]}")
        
        # Create new BLEU metric instance
        bleu_metric = BLEUScore()
        bleu = bleu_metric(tokenized_predicted, references)
        
        print_msg(f"BLEU score: {bleu.item():.6f}")
        metrics['bleu'] = bleu.item()
        
        # Log to TensorBoard if writer exists
        if writer is not None:
            log_to_tensorboard(writer, 'validation/BLEU', bleu, global_step, flush=True)
            
    except Exception as e:
        print_msg(f"Error calculating BLEU score: {str(e)}")
        # As a fallback, try another approach
        try:
            # Use a different approach - calculate BLEU without tokenization
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            
            print_msg("Trying NLTK corpus_bleu as fallback")
            
            # Format for NLTK's corpus_bleu
            references = [[r.split()] for r in expected]  # Tokenize again
            hypotheses = [p.split() for p in predicted]   # Tokenize again
            
            # Use smoothing
            smoothie = SmoothingFunction().method3
            bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
            
            print_msg(f"NLTK BLEU score: {bleu:.6f}")
            metrics['bleu'] = bleu
            
            # Log to TensorBoard if writer exists
            if writer is not None:
                log_to_tensorboard(writer, 'validation/BLEU', bleu, global_step, flush=True)
                
        except Exception as e2:
            print_msg(f"Fallback BLEU calculation also failed: {str(e2)}")
            metrics['bleu'] = 0.0

    # Flush writer at the end
    if writer is not None:
        writer.flush()

    return metrics

def get_ds(config, tokenizer_src=None, tokenizer_tgt=None, is_distributed=False, rank=0, world_size=1):
    # Load tokenizers using SentencePiece if not provided
    if tokenizer_src is None or tokenizer_tgt is None:
        tokenizer_src, tokenizer_tgt = spm.SentencePieceProcessor(), spm.SentencePieceProcessor()
        tokenizer_src.load(config['tokenizer_src_path'])
        tokenizer_tgt.load(config['tokenizer_tgt_path'])
    
    # Get word dropout rate from config
    word_dropout_rate = config.get('word_dropout_rate', 0.1)
    
    # Define the train and validation datasets
    train_ds = BilingualDataset(
        config['train_src_file'],
        config['train_tgt_file'],
        tokenizer_src,
        tokenizer_tgt,
        max_len=config.get('max_len', 500),
        word_dropout_rate=word_dropout_rate,
        is_train=True
    )
    
    val_ds = BilingualDataset(
        config['val_src_file'],
        config['val_tgt_file'],
        tokenizer_src,
        tokenizer_tgt,
        max_len=config.get('max_len', 500),
        word_dropout_rate=0.0,  # No word dropout for validation
        is_train=False
    )

    # Create collate function with the tokenizers
    train_collate_fn = functools.partial(collate_batch, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt)
    val_collate_fn = functools.partial(collate_batch, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt)

    # Calculate max lengths if needed (only on rank 0 for distributed)
    if not is_distributed or rank == 0:
        max_len_src = 0
        max_len_tgt = 0

        # Sample a small subset to find max lengths (optional)
        sample_size = min(100, len(train_ds))
        for i in range(sample_size):
            src_tokens = train_ds[i]["src_tokens"]
            tgt_tokens = train_ds[i]["tgt_tokens"]
            max_len_src = max(max_len_src, len(src_tokens))
            max_len_tgt = max(max_len_tgt, len(tgt_tokens))

        print(f'Max length of source sentence (sample): {max_len_src}')
        print(f'Max length of target sentence (sample): {max_len_tgt}')

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
    
    # Pass the collate_fn to DataLoader
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        collate_fn=train_collate_fn,
        pin_memory=True,
        num_workers=2,
    )
    
    val_dataloader = DataLoader(
        val_ds, 
        batch_size=1,  # Always use batch size 1 for validation
        shuffle=False,
        sampler=val_sampler,
        collate_fn=val_collate_fn,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, train_sampler

def get_model(config, vocab_src_len, vocab_tgt_len, device=None):
    # Use config parameters for model construction
    d_model = config.get('d_model', 384)
    num_layers = config.get('num_layers', 4)
    num_heads = config.get('num_heads', 8)
    dropout = config.get('dropout', 0.15)
    d_ff = config.get('d_ff', 1536)
    
    # Use max_len for maximum sequence length
    max_len = config.get('max_len', 500)
    
    # Build transformer with specified parameters
    model = build_transformer(
        vocab_src_len, 
        vocab_tgt_len, 
        max_len, 
        max_len, 
        d_model=d_model,
        N=num_layers,
        h=num_heads,
        dropout=dropout,
        d_ff=d_ff
    )
    
    # Move to specified device if provided
    if device is not None:
        model = model.to(device)
        
    return model

def get_autocast_context(use_mixed_precision=False):
    """Helper function to get the right autocast context manager"""
    # Import only if needed
    if use_mixed_precision:
        from torch.amp import autocast
        return autocast(device_type='cuda')
    else:
        # Return a dummy context manager that does nothing
        from contextlib import nullcontext
        return nullcontext()

def log_to_tensorboard(writer, tag, value, step, flush=False):
    """Helper function to consistently log metrics to TensorBoard
    
    Args:
        writer: The TensorBoard SummaryWriter
        tag: The tag/name for the metric
        value: The value to log (will extract .item() if it's a tensor)
        step: The global step
        flush: Whether to flush after logging
    """
    if writer is not None:
        # Extract .item() from tensors to avoid potential CUDA synchronization issues
        if hasattr(value, 'item'):
            value = value.item()
        
        writer.add_scalar(tag, value, step)
        
        if flush:
            writer.flush()

def train_model_distributed(rank, world_size, config):
    """Train the model in distributed mode.
    Args:
        rank: Current process rank
        world_size: Total number of processes
        config: Training configuration
    """
    # Set up distributed training environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend=config.get('backend', 'nccl'),
        rank=rank,
        world_size=world_size
    )
    
    # Get device for current process
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print(f"Training with distributed data parallel on {world_size} GPUs")
    
    # Set device specific seed for reproducibility
    torch.manual_seed(42 + rank)
    
    # Make sure the weights folder exists on the main process
    if rank == 0:
        Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    # Wait for the main process to create directories
    dist.barrier()
    
    # Get dataloaders, tokenziers, and samplers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, train_sampler = get_ds(
        config, 
        is_distributed=True, 
        rank=rank, 
        world_size=world_size
    )
    
    # Create model and move to correct device
    model = get_model(config, tokenizer_src.vocab_size(), tokenizer_tgt.vocab_size(), device)
    
    # Wrap model in DDP
    model = DDP(
        model, 
        device_ids=[rank] if torch.cuda.is_available() else None,
        output_device=rank if torch.cuda.is_available() else None,
        find_unused_parameters=config.get('find_unused_parameters', False)
    )
    
    # Initialize TensorBoard writer (only on main process)
    writer = None
    if rank == 0:
        writer = SummaryWriter(config['experiment_name'])

    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        eps=1e-9,
        weight_decay=config.get('weight_decay', 0.01)
    )

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    
    if preload and rank == 0:
        # Print the preload value for debugging
        if rank == 0:
            print(f'Preload value: {preload}')
        
        # Determine if we should load a model
        if os.path.exists(preload):
            # Use the provided path directly
            model_filename = preload
            if rank == 0:
                print(f'Using model path: {model_filename}')
                print(f'Model file exists: {os.path.exists(model_filename)}')
            
            # Load the model
            if rank == 0:
                print(f'Loading model from {model_filename}')
            state = torch.load(model_filename, map_location=device)
            model.module.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state.get('global_step', 0)
            if rank == 0:
                print(f'Loaded checkpoint. Resuming from epoch {initial_epoch}, global step {global_step}')
        else:
            if rank == 0:
                print(f'Model file not found at {preload}')
    else:
        if rank == 0:
            print('No model to preload, starting from scratch')
            
    # Broadcast initial_epoch and global_step from rank 0 to all processes
    if world_size > 1:
        initial_epoch_tensor = torch.tensor(initial_epoch, device=device)
        global_step_tensor = torch.tensor(global_step, device=device)
        
        # Broadcast tensors to all processes
        dist.broadcast(initial_epoch_tensor, 0)
        dist.broadcast(global_step_tensor, 0)
        
        # Update values from received broadcasts
        initial_epoch = initial_epoch_tensor.item()
        global_step = global_step_tensor.item()
    
    # Set up learning rate scheduler with warmup
    total_steps = len(train_dataloader) * config['num_epochs']
    warmup_steps = config.get('warmup_steps', 4000)
    
    def lr_lambda(step):
        # Linear warmup followed by cosine decay
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine annealing after warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, (total_steps - warmup_steps))))
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    # SentencePiece uses piece_to_id instead of token_to_id
    pad_idx = tokenizer_src.piece_to_id("<pad>")
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_idx, 
        label_smoothing=config.get('label_smoothing', 0.1)
    ).to(device)

    # Mixed precision setup
    use_mixed_precision = config.get('use_mixed_precision', True) and device.type == 'cuda'
    
    # Set up autocast and GradScaler for mixed precision
    if use_mixed_precision:
        from torch.amp import autocast, GradScaler
        amp_context = autocast(device_type='cuda')
        scaler = GradScaler()
        if rank == 0:
            print("Using mixed precision training")
    else:
        from contextlib import nullcontext
        amp_context = nullcontext()
        scaler = None

    # Best model tracking (even if early stopping is disabled)
    monitored_metric = config.get('early_stopping_metric', 'bleu')
    best_metric_value = float('-inf') if monitored_metric in ['bleu'] else float('inf')
    patience_counter = 0
    
    print(f"Tracking best model based on {monitored_metric} metric")
    print(f"Early stopping: {'enabled' if config.get('early_stopping', False) else 'disabled'}")
    if config.get('early_stopping', False):
        print(f"Patience: {config.get('early_stopping_patience', 10)} epochs")
    
    # Model saving strategy
    save_only_best = config.get('save_only_best', False)
    print(f"Model saving strategy: {'Only best models' if save_only_best else 'Best + epoch models'}")
    if save_only_best:
        print("  - Only the latest best model will be saved")
        print("  - Previous best models will be automatically removed")
    else:
        print("  - Both best models and epoch models will be saved")
        print(f"  - Maximum checkpoints to keep: {config.get('keep_checkpoint_max', 'unlimited')}")
    
    # Gradient accumulation
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {config['batch_size'] * grad_accum_steps}")
    
    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        # Set epoch for sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Set model to training mode
        model.train()
        
        # Only create progress bar on rank 0
        if rank == 0:
            progress = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        else:
            progress = train_dataloader
        
        # Training loss for this epoch
        total_loss = 0.0
        batch_count = 0
        
        # Reset optimizer gradients at the start of each epoch
        optimizer.zero_grad(set_to_none=True)
        
        for i, batch in enumerate(progress):
            # Track batch count
            batch_count += 1
            
            # Move batch to device
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Calculate loss with gradient accumulation
            # Normalize loss by grad_accum_steps
            loss_factor = 1.0 / grad_accum_steps

            # Forward pass with autocast for mixed precision
            with amp_context:
                encoder_output = model.module.encode(encoder_input, encoder_mask)
                decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.module.project(decoder_output)
                
                # Compute loss
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.vocab_size()), label.view(-1)) * loss_factor
            
            # Backward pass with gradient scaling if using mixed precision
            if use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if we reached the accumulation steps
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_dataloader):
                if use_mixed_precision:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_val', 1.0))
                
                if use_mixed_precision:
                    # Update weights with scaled gradients
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Update weights
                    optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Reset gradients
                optimizer.zero_grad(set_to_none=True)
            
            # Accumulate full loss for logging (not the scaled loss)
            full_loss = loss.item() * grad_accum_steps
            total_loss += full_loss
            
            # Update progress bar on rank 0
            if rank == 0:
                progress.set_postfix({
                    "loss": f"{full_loss:6.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}"
                })
            
            # Log metrics on main process
            if rank == 0 and writer is not None:
                log_to_tensorboard(writer, 'train/loss', full_loss, global_step)
                log_to_tensorboard(writer, 'train/learning_rate', scheduler.get_last_lr()[0], global_step, flush=True)
            
            global_step += 1
            
            # Run validation if needed based on steps
            if rank == 0 and config.get('evaluation_strategy', 'epoch') == 'steps' and global_step % config.get('eval_steps', 1000) == 0:
                # We only run validation on the main process
                model.eval()
                max_len_val = config.get('max_len', 500)
                metrics = run_validation(
                    model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                    max_len_val, device, lambda msg: print(msg) if rank == 0 else None, 
                    global_step, writer,
                    num_examples=5,  # Hiển thị nhiều ví dụ hơn
                    num_display=5    # Hiển thị tất cả các ví dụ đánh giá
                )
                model.train()  # Switch back to train mode
                
                # Save checkpoint at this step if needed
                if config.get('save_strategy', 'epoch') == 'steps' and global_step % config.get('save_steps', 1000) == 0:
                    checkpoint_filename = get_weights_file_path(config, f"step_{global_step}")
                    # Save on main process only
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.module.state_dict(),  # Save the inner model, not DDP wrapper
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'metrics': metrics
                    }, checkpoint_filename)
                    print(f"Rank {rank}: Saved checkpoint at step {global_step} to {checkpoint_filename}")
                
                # Apply early stopping logic
                best_metric_value, patience_counter, should_stop = _apply_early_stopping(
                    config, metrics, best_metric_value, patience_counter, global_step,
                    model.module, optimizer, scheduler, epoch
                )
        
        # Calculate average training loss for this epoch
        # Gather losses from all GPUs
        if world_size > 1:
            # Create tensor with local loss sum
            loss_tensor = torch.tensor([total_loss], device=device)
            # Create output tensor
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            # Get the global loss
            total_loss = loss_tensor.item()
            
            # Same for batch count
            count_tensor = torch.tensor([batch_count], device=device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            batch_count = count_tensor.item()
        
        avg_train_loss = total_loss / max(1, batch_count)
        
        if rank == 0:
            print(f"Epoch {epoch:02d} - Average training loss: {avg_train_loss:.4f}")
            log_to_tensorboard(writer, 'train/average_loss', avg_train_loss, epoch, flush=True)

        # Run validation at the end of every epoch if strategy is epoch
        if rank == 0 and config.get('evaluation_strategy', 'epoch') == 'epoch':
            model.eval()
            max_len_val = config.get('max_len', 500)
            
            metrics = run_validation(
                model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                max_len_val, device, lambda msg: print(msg) if rank == 0 else None, 
                global_step, writer,
                num_examples=5,  # Hiển thị nhiều ví dụ hơn
                num_display=5    # Hiển thị tất cả các ví dụ đánh giá
            )
            
            # In kết quả BLEU và các metrics khác
            if metrics:
                print("-" * 80)
                print(f"Validation results at epoch {epoch}:")
                for metric_name, metric_value in metrics.items():
                    print(f"  {metric_name}: {metric_value:.6f}")
                print("-" * 80)
            
            model.train()  # Switch back to train mode

            # Save the model at the end of every epoch if strategy is epoch
            if config.get('save_strategy', 'epoch') == 'epoch' and not config.get('save_only_best', False):
                model_filename = get_weights_file_path(config, f"{epoch:02d}")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),  # Save the inner model, not DDP wrapper
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'metrics': metrics
                }, model_filename)
                print(f"Rank {rank}: Saved model at epoch {epoch} to {model_filename}")
            elif config.get('save_only_best', False):
                print(f"Epoch {epoch}: Only saving best models (save_only_best=True)")
            
            # Apply early stopping logic
            should_stop = _apply_early_stopping(
                config, metrics, best_metric_value, patience_counter, global_step,
                model.module, optimizer, scheduler, epoch
            )
            
            if should_stop:
                print(f"Early stopping triggered! No improvement for {patience_counter} epochs.")
                break
        
        # Synchronize all processes before starting next epoch
        if world_size > 1:
            dist.barrier()
    
    # Cleanup distributed processes
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        print("Training completed!")
        if best_metric_value and os.path.exists(get_weights_file_path(config, 'best')):
            print(f"Best validation {monitored_metric}: {best_metric_value:.6f}")
            
        # Properly close the TensorBoard writer
        if writer is not None:
            writer.close()

def _apply_early_stopping(config, metrics, best_metric_value, patience_counter, global_step,
                          model, optimizer, scheduler, epoch):
    """Helper function to apply early stopping logic and save the best model"""
    should_stop = False
    is_best_model = False
    monitored_metric = config.get('early_stopping_metric', 'bleu')
    
    if metrics:
        # Get the value of the monitored metric
        current_metric_value = metrics.get(monitored_metric)
        
        if current_metric_value is not None:
            # Check if this is a new best model
            is_improvement = False
            
            # For metrics where higher is better (like BLEU)
            if monitored_metric in ['bleu']:
                is_improvement = current_metric_value > (best_metric_value + config.get('early_stopping_min_delta', 0.0001))
            # For metrics where lower is better (like loss, WER, CER)
            else:
                is_improvement = current_metric_value < (best_metric_value - config.get('early_stopping_min_delta', 0.0001))
            
            if is_improvement:
                print(f"Validation {monitored_metric} improved from {best_metric_value:.6f} to {current_metric_value:.6f}")
                # Update best metric value
                best_metric_value = current_metric_value
                # Reset patience counter
                patience_counter = 0
                # Flag this as best model
                is_best_model = True
                
                # Get the path for the best model
                best_model_filename = get_weights_file_path(config, "best")
                
                # Remove old best model if it exists
                if os.path.exists(best_model_filename):
                    try:
                        os.remove(best_model_filename)
                        print(f"Removed old best model: {best_model_filename}")
                    except Exception as e:
                        print(f"Warning: Could not remove old best model: {e}")
                
                # Save the new best model
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'metrics': metrics,
                    'best_metric': {monitored_metric: current_metric_value}
                }, best_model_filename)
                print(f"Saved new best model to {best_model_filename}")
                
            else:
                # Increment patience counter if early stopping is enabled
                if config.get('early_stopping', False):
                    new_patience = patience_counter + 1
                    patience_counter = new_patience
                    print(f"Validation {monitored_metric} did not improve. Patience: {new_patience}/{config.get('early_stopping_patience', 10)}")
                    
                    if new_patience >= config.get('early_stopping_patience', 10):
                        should_stop = True
                        print(f"Early stopping triggered! No improvement for {new_patience} epochs.")
    
    return best_metric_value, patience_counter, should_stop

def train_model_dataparallel(config, device_ids=None):
    """Train model using DataParallel on multiple GPUs
    
    Args:
        config: Training configuration
        device_ids: List of GPU device IDs to use (e.g., [0, 1])
    """
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    print(f"Using DataParallel training on GPUs: {device_ids}")
    
    # Primary device is the first GPU
    device = torch.device(f'cuda:{device_ids[0]}')
    
    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    # Get datasets and tokenizers (no distributed sampler needed for DataParallel)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, _ = get_ds(
        config, 
        is_distributed=False,  # DataParallel doesn't need distributed sampling
        rank=0, 
        world_size=1
    )
    
    # Create model and move to primary device
    model = get_model(config, tokenizer_src.vocab_size(), tokenizer_tgt.vocab_size(), device)
    
    # Wrap model in DataParallel
    if len(device_ids) > 1:
        model = DP(model, device_ids=device_ids)
        print(f"Model wrapped in DataParallel with devices: {device_ids}")
    else:
        print("Only one GPU specified, using single GPU training")
    
    # TensorBoard writer
    writer = SummaryWriter(config['experiment_name'])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        eps=1e-9,
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Preload model if requested
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    
    print(f'Preload value: {preload}')
    
    if preload and os.path.exists(preload):
        model_filename = preload
        print(f'Loading model from {model_filename}')
        
        state = torch.load(model_filename, map_location=device)
        
        # Handle loading state dict for DataParallel models
        if isinstance(model, DP):
            # If the saved model was not DataParallel, we need to load into model.module
            if any(key.startswith('module.') for key in state['model_state_dict'].keys()):
                model.load_state_dict(state['model_state_dict'])
            else:
                # Add 'module.' prefix to keys
                new_state_dict = {}
                for key, value in state['model_state_dict'].items():
                    new_state_dict[f'module.{key}'] = value
                model.load_state_dict(new_state_dict)
        else:
            # Regular model loading
            if any(key.startswith('module.') for key in state['model_state_dict'].keys()):
                # Remove 'module.' prefix from keys
                new_state_dict = {}
                for key, value in state['model_state_dict'].items():
                    if key.startswith('module.'):
                        new_state_dict[key[7:]] = value  # Remove 'module.' prefix
                    else:
                        new_state_dict[key] = value
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state['model_state_dict'])
        
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state.get('global_step', 0)
        print(f'Loaded checkpoint. Resuming from epoch {initial_epoch}, global step {global_step}')
    else:
        print('No model to preload, starting from scratch')
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * config['num_epochs']
    warmup_steps = config.get('warmup_steps', 4000)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, (total_steps - warmup_steps))))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    pad_idx = tokenizer_src.piece_to_id("<pad>")
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_idx, 
        label_smoothing=config.get('label_smoothing', 0.1)
    ).to(device)
    
    # Mixed precision setup
    use_mixed_precision = config.get('use_mixed_precision', True) and device.type == 'cuda'
    
    if use_mixed_precision:
        from torch.amp import autocast, GradScaler
        amp_context = autocast(device_type='cuda')
        scaler = GradScaler()
        print("Using mixed precision training")
    else:
        from contextlib import nullcontext
        amp_context = nullcontext()
        scaler = None
    
    # Best model tracking
    monitored_metric = config.get('early_stopping_metric', 'bleu')
    best_metric_value = float('-inf') if monitored_metric in ['bleu'] else float('inf')
    patience_counter = 0
    
    print(f"Tracking best model based on {monitored_metric} metric")
    print(f"Early stopping: {'enabled' if config.get('early_stopping', False) else 'disabled'}")
    if config.get('early_stopping', False):
        print(f"Patience: {config.get('early_stopping_patience', 10)} epochs")
    
    # Model saving strategy
    save_only_best = config.get('save_only_best', False)
    print(f"Model saving strategy: {'Only best models' if save_only_best else 'Best + epoch models'}")
    
    # Gradient accumulation
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {config['batch_size'] * grad_accum_steps}")
    
    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Set model to training mode
        model.train()
        
        # Create progress bar
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        
        # Training metrics
        total_loss = 0.0
        batch_count = 0
        
        # Reset optimizer gradients
        optimizer.zero_grad(set_to_none=True)
        
        for i, batch in enumerate(progress):
            batch_count += 1
            
            # Move batch to device
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Calculate loss with gradient accumulation
            loss_factor = 1.0 / grad_accum_steps
            
            # Forward pass with autocast for mixed precision
            with amp_context:
                # Handle DataParallel model access
                if isinstance(model, DP):
                    encoder_output = model.module.encode(encoder_input, encoder_mask)
                    decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.module.project(decoder_output)
                else:
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.project(decoder_output)
                
                # Compute loss
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.vocab_size()), label.view(-1)) * loss_factor
            
            # Backward pass with gradient scaling if using mixed precision
            if use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if we reached the accumulation steps
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_dataloader):
                if use_mixed_precision:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_val', 1.0))
                
                if use_mixed_precision:
                    # Update weights with scaled gradients
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Update weights
                    optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Reset gradients
                optimizer.zero_grad(set_to_none=True)
            
            # Accumulate full loss for logging
            full_loss = loss.item() * grad_accum_steps
            total_loss += full_loss
            
            # Update progress bar
            progress.set_postfix({
                "loss": f"{full_loss:6.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"
            })
            
            # Log metrics
            log_to_tensorboard(writer, 'train/loss', full_loss, global_step)
            log_to_tensorboard(writer, 'train/learning_rate', scheduler.get_last_lr()[0], global_step, flush=True)
            
            global_step += 1
            
            # Run validation if needed based on steps
            if config.get('evaluation_strategy', 'epoch') == 'steps' and global_step % config.get('eval_steps', 1000) == 0:
                model.eval()
                max_len_val = config.get('max_len', 500)
                
                metrics = run_validation(
                    model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                    max_len_val, device, lambda msg: print(msg), 
                    global_step, writer,
                    num_examples=5,
                    num_display=5
                )
                
                if metrics:
                    print("-" * 80)
                    print(f"Validation results at step {global_step}:")
                    for metric_name, metric_value in metrics.items():
                        print(f"  {metric_name}: {metric_value:.6f}")
                    print("-" * 80)
                
                model.train()  # Switch back to train mode
                
                # Check for best model and apply early stopping logic
                actual_model = model.module if isinstance(model, DP) else model
                best_metric_value, patience_counter, should_stop = _apply_early_stopping(
                    config, metrics, best_metric_value, patience_counter, global_step,
                    actual_model, optimizer, scheduler, epoch
                )
                
                if should_stop:
                    break
        
        # Calculate average training loss for this epoch
        avg_train_loss = total_loss / batch_count
        print(f"Epoch {epoch:02d} - Average training loss: {avg_train_loss:.4f}")
        log_to_tensorboard(writer, 'train/average_loss', avg_train_loss, epoch, flush=True)

        # Run validation at the end of every epoch
        if config.get('evaluation_strategy', 'epoch') == 'epoch':
            model.eval()
            max_len_val = config.get('max_len', 500)
            
            metrics = run_validation(
                model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                max_len_val, device, lambda msg: print(msg), 
                global_step, writer,
                num_examples=5,
                num_display=5
            )
            
            if metrics:
                print("-" * 80)
                print(f"Validation results at epoch {epoch}:")
                for metric_name, metric_value in metrics.items():
                    print(f"  {metric_name}: {metric_value:.6f}")
                print("-" * 80)
            
            model.train()  # Switch back to train mode
            
            # Save the model at the end of every epoch if strategy is epoch
            if config.get('save_strategy', 'epoch') == 'epoch' and not config.get('save_only_best', False):
                model_filename = get_weights_file_path(config, f"{epoch:02d}")
                # Get the actual model state dict
                actual_model = model.module if isinstance(model, DP) else model
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': actual_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'metrics': metrics
                }, model_filename)
                print(f"Saved model at epoch {epoch} to {model_filename}")
            
            # Check for best model and apply early stopping logic
            actual_model = model.module if isinstance(model, DP) else model
            best_metric_value, patience_counter, should_stop = _apply_early_stopping(
                config, metrics, best_metric_value, patience_counter, global_step,
                actual_model, optimizer, scheduler, epoch
            )
            
            if should_stop:
                print(f"Early stopping triggered! No improvement for {patience_counter} epochs.")
                break
    
    print("Training completed!")
    print(f"Best validation {monitored_metric}: {best_metric_value:.6f}")
    print(f"Best model saved at: {get_weights_file_path(config, 'best')}")
    
    # Properly close the TensorBoard writer
    if writer is not None:
        writer.close()

def train_model(config=None):
    """Main entry point for training"""
    if config is None:
        config = get_config()
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Check if we're running on Kaggle
    is_kaggle = os.path.exists('/kaggle')
    
    # Get training mode preference
    training_mode = config.get('training_mode', 'auto')  # 'auto', 'single', 'dataparallel', 'distributed'
    force_single_gpu = config.get('force_single_gpu', False)
    
    print(f"Available GPUs: {num_gpus}")
    print(f"Training mode: {training_mode}")
    print(f"Kaggle environment: {is_kaggle}")
    
    if force_single_gpu or num_gpus <= 1:
        print("Using single GPU/CPU training")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_model_single(config, device)
        return
    
    # Multi-GPU training logic
    if training_mode == 'dataparallel' or (training_mode == 'auto' and is_kaggle):
        # Use DataParallel for Kaggle or when explicitly requested
        device_ids = list(range(min(num_gpus, config.get('num_gpus', num_gpus))))
        print(f"Using DataParallel training on GPUs: {device_ids}")
        train_model_dataparallel(config, device_ids)
        return
    
    elif training_mode == 'distributed' or (training_mode == 'auto' and not is_kaggle):
        # Use DistributedDataParallel for non-Kaggle environments or when explicitly requested
        use_distributed = config.get('distributed_training', True)
        world_size = min(config.get('num_gpus', num_gpus), num_gpus)
        
        if use_distributed and world_size > 1:
            try:
                print(f"Attempting DistributedDataParallel training on {world_size} GPUs...")
                
                # Set environment variables
                os.environ['NCCL_TIMEOUT'] = str(config.get('nccl_timeout', 3600000))
                
                # Try distributed training
                mp.start_method = 'spawn'
                process_context = mp.spawn(
                    train_model_distributed,
                    args=(world_size, config),
                    nprocs=world_size,
                    join=False
                )
                
                # Wait for completion with timeout check
                start_time = time.time()
                early_failure_timeout = 60
                
                while time.time() - start_time < early_failure_timeout:
                    if not all(process.is_alive() for process in process_context.processes):
                        raise Exception("One of the distributed processes failed to start properly")
                    time.sleep(1)
                
                process_context.join()
                print("DistributedDataParallel training completed successfully!")
                return
                
            except Exception as e:
                print(f"DistributedDataParallel training failed: {str(e)}")
                print("Falling back to DataParallel training...")
                device_ids = list(range(min(num_gpus, config.get('num_gpus', num_gpus))))
                train_model_dataparallel(config, device_ids)
                return
    
    # Fallback to single GPU training
    print("Falling back to single GPU training")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_model_single(config, device)

def train_model_single(config, device):
    """Train model on a single device (CPU or one GPU)"""
    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    # Get datasets and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, _ = get_ds(
        config, 
        is_distributed=False,
        rank=0, 
        world_size=1
    )
    
    # Create model and move to device
    model = get_model(config, tokenizer_src.vocab_size(), tokenizer_tgt.vocab_size(), device)
    
    # TensorBoard writer
    writer = SummaryWriter(config['experiment_name'])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        eps=1e-9,
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Preload model if requested
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    
    # Print the preload value for debugging
    print(f'Preload value: {preload}')
    
    # Determine if we should load a model
    if preload and os.path.exists(preload):
        # Use the provided path directly
        model_filename = preload
        print(f'Using model path: {model_filename}')
        
        # Check if file exists
        file_exists = os.path.exists(model_filename)
        print(f'Model file exists: {file_exists}')
        
        # Load the model
        if file_exists:
            print(f'Loading model from {model_filename}')
            state = torch.load(model_filename, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state.get('global_step', 0)
            print(f'Loaded checkpoint. Resuming from epoch {initial_epoch}, global step {global_step}')
        else:
            print(f'Model file not found at {model_filename}')
    else:
        print('No model to preload, starting from scratch')
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * config['num_epochs']
    warmup_steps = config.get('warmup_steps', 4000)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, (total_steps - warmup_steps))))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    pad_idx = tokenizer_src.piece_to_id("<pad>")
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_idx, 
        label_smoothing=config.get('label_smoothing', 0.1)
    ).to(device)
    
    # Mixed precision setup
    use_mixed_precision = config.get('use_mixed_precision', True) and device.type == 'cuda'
    
    # Set up autocast and GradScaler for mixed precision
    if use_mixed_precision:
        from torch.amp import autocast, GradScaler
        amp_context = autocast(device_type='cuda')
        scaler = GradScaler()
        print("Using mixed precision training")
    else:
        from contextlib import nullcontext
        amp_context = nullcontext()
        scaler = None
    
    # Best model tracking (even if early stopping is disabled)
    monitored_metric = config.get('early_stopping_metric', 'bleu')
    best_metric_value = float('-inf') if monitored_metric in ['bleu'] else float('inf')
    patience_counter = 0
    
    print(f"Tracking best model based on {monitored_metric} metric")
    print(f"Early stopping: {'enabled' if config.get('early_stopping', False) else 'disabled'}")
    if config.get('early_stopping', False):
        print(f"Patience: {config.get('early_stopping_patience', 10)} epochs")
    
    # Model saving strategy
    save_only_best = config.get('save_only_best', False)
    print(f"Model saving strategy: {'Only best models' if save_only_best else 'Best + epoch models'}")
    if save_only_best:
        print("  - Only the latest best model will be saved")
        print("  - Previous best models will be automatically removed")
    else:
        print("  - Both best models and epoch models will be saved")
        print(f"  - Maximum checkpoints to keep: {config.get('keep_checkpoint_max', 'unlimited')}")
    
    # Gradient accumulation
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {config['batch_size'] * grad_accum_steps}")
    
    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Set model to training mode
        model.train()
        
        # Create progress bar
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        
        # Training metrics
        total_loss = 0.0
        batch_count = 0
        
        # Reset optimizer gradients
        optimizer.zero_grad(set_to_none=True)
        
        for i, batch in enumerate(progress):
            batch_count += 1
            
            # Move batch to device
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Calculate loss with gradient accumulation
            loss_factor = 1.0 / grad_accum_steps
            
            # Forward pass with autocast for mixed precision
            with amp_context:
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                
                # Compute loss
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.vocab_size()), label.view(-1)) * loss_factor
            
            # Backward pass with gradient scaling if using mixed precision
            if use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if we reached the accumulation steps
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_dataloader):
                if use_mixed_precision:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_val', 1.0))
                
                if use_mixed_precision:
                    # Update weights with scaled gradients
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Update weights
                    optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Reset gradients
                optimizer.zero_grad(set_to_none=True)
            
            # Accumulate full loss for logging
            full_loss = loss.item() * grad_accum_steps
            total_loss += full_loss
            
            # Update progress bar
            progress.set_postfix({
                "loss": f"{full_loss:6.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"
            })
            
            # Log metrics
            log_to_tensorboard(writer, 'train/loss', full_loss, global_step)
            log_to_tensorboard(writer, 'train/learning_rate', scheduler.get_last_lr()[0], global_step, flush=True)
            
            global_step += 1
            
            # Run validation if needed based on steps
            if config.get('evaluation_strategy', 'epoch') == 'steps' and global_step % config.get('eval_steps', 1000) == 0:
                model.eval()
                max_len_val = config.get('max_len', 500)
                
                metrics = run_validation(
                    model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                    max_len_val, device, lambda msg: print(msg), 
                    global_step, writer,
                    num_examples=5,
                    num_display=5
                )
                
                # Print validation results
                if metrics:
                    print("-" * 80)
                    print(f"Validation results at epoch {epoch}:")
                    for metric_name, metric_value in metrics.items():
                        print(f"  {metric_name}: {metric_value:.6f}")
                    print("-" * 80)
                
                model.train()  # Switch back to train mode
                
                # Check for best model and apply early stopping logic
                best_metric_value, patience_counter, should_stop = _apply_early_stopping(
                    config, metrics, best_metric_value, patience_counter, global_step,
                    model, optimizer, scheduler, epoch
                )
                
                if should_stop:
                    break
        
        # Calculate average training loss for this epoch
        avg_train_loss = total_loss / batch_count
        print(f"Epoch {epoch:02d} - Average training loss: {avg_train_loss:.4f}")
        log_to_tensorboard(writer, 'train/average_loss', avg_train_loss, epoch, flush=True)

        # Run validation at the end of every epoch
        model.eval()
        max_len_val = config.get('max_len', 500)
        
        metrics = run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, 
            max_len_val, device, lambda msg: print(msg), 
            global_step, writer,
            num_examples=5,
            num_display=5
        )
        
        # Print validation results
        if metrics:
            print("-" * 80)
            print(f"Validation results at epoch {epoch}:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.6f}")
            print("-" * 80)
        
        model.train()  # Switch back to train mode
        
        # Check for best model and apply early stopping logic
        best_metric_value, patience_counter, should_stop = _apply_early_stopping(
            config, metrics, best_metric_value, patience_counter, global_step,
            model, optimizer, scheduler, epoch
        )
        
        if should_stop:
            break
    
    print("Training completed!")
    print(f"Best validation {monitored_metric}: {best_metric_value:.6f}")
    print(f"Best model saved at: {get_weights_file_path(config, 'best')}")
    
    # Properly close the TensorBoard writer
    if writer is not None:
        writer.close()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
