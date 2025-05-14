import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import sentencepiece as spm

class BilingualDataset(Dataset):

    def __init__(self, src_file, tgt_file, tokenizer_src, tokenizer_tgt, max_len=None):
        super().__init__()
        self.max_len = max_len  # Optional max length for filtering

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        
        # Load the source and target texts
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_texts = f.read().splitlines()
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_texts = f.read().splitlines()
            
        # Ensure src and tgt have the same number of sentences
        assert len(self.src_texts) == len(self.tgt_texts), "Source and target files must have the same number of lines"

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Transform the text into tokens using SentencePiece
        enc_input_tokens = self.tokenizer_src.encode(src_text)
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text)
        
        # Optional filtering of sequences that are too long
        if self.max_len is not None:
            if len(enc_input_tokens) > self.max_len - 2 or len(dec_input_tokens) > self.max_len - 2:
                # Skip this example and get another one
                idx = (idx + 1) % len(self)
                return self[idx]

        return {
            "src_tokens": enc_input_tokens,
            "tgt_tokens": dec_input_tokens,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    # Create a causal mask of shape [1, size, size]
    # The mask is 1 where attention is allowed, 0 where it is not
    mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.int)
    return (mask == 0).unsqueeze(0)  # [1, size, size], True where allowed

def collate_batch(batch, tokenizer_src, tokenizer_tgt):
    # Get special token IDs
    sos_token_src = tokenizer_src.piece_to_id("<s>")
    eos_token_src = tokenizer_src.piece_to_id("</s>")
    pad_token_src = tokenizer_src.piece_to_id("<pad>")
    
    sos_token_tgt = tokenizer_tgt.piece_to_id("<s>")
    eos_token_tgt = tokenizer_tgt.piece_to_id("</s>")
    pad_token_tgt = tokenizer_tgt.piece_to_id("<pad>")
    
    # Get max lengths in this batch
    max_src_len = max([len(item["src_tokens"]) for item in batch]) + 2  # +2 for SOS and EOS
    max_tgt_len = max([len(item["tgt_tokens"]) for item in batch]) + 1  # +1 for SOS (EOS is in labels)
    
    # Initialize tensors for the batch
    batch_size = len(batch)
    encoder_inputs = torch.full((batch_size, max_src_len), pad_token_src, dtype=torch.int64)
    decoder_inputs = torch.full((batch_size, max_tgt_len), pad_token_tgt, dtype=torch.int64)
    labels = torch.full((batch_size, max_tgt_len), pad_token_tgt, dtype=torch.int64)
    
    # Prepare encoder/decoder masks (initialized to all zeros = masked)
    encoder_masks = torch.zeros((batch_size, 1, 1, max_src_len), dtype=torch.int64)
    
    # Process each item in the batch
    src_texts = []
    tgt_texts = []
    
    for i, item in enumerate(batch):
        src_tokens = item["src_tokens"]
        tgt_tokens = item["tgt_tokens"]
        src_texts.append(item["src_text"])
        tgt_texts.append(item["tgt_text"])
        
        # Add SOS and EOS to source
        src_length = len(src_tokens) + 2
        encoder_inputs[i, 0] = sos_token_src
        encoder_inputs[i, 1:src_length-1] = torch.tensor(src_tokens, dtype=torch.int64)
        encoder_inputs[i, src_length-1] = eos_token_src
        
        # Create encoder mask (1 for tokens, 0 for padding)
        encoder_masks[i, 0, 0, :src_length] = 1
        
        # Add SOS to decoder input
        tgt_length = len(tgt_tokens) + 1
        decoder_inputs[i, 0] = sos_token_tgt
        decoder_inputs[i, 1:tgt_length] = torch.tensor(tgt_tokens, dtype=torch.int64)
        
        # Add target tokens and EOS to labels
        labels[i, :tgt_length-1] = torch.tensor(tgt_tokens, dtype=torch.int64)
        labels[i, tgt_length-1] = eos_token_tgt
    
    # Create decoder masks (combining padding mask with causal mask)
    decoder_padding_masks = (decoder_inputs != pad_token_tgt).unsqueeze(1).int()  # [batch, 1, tgt_len]
    causal_masks = causal_mask(max_tgt_len)  # [1, tgt_len, tgt_len]
    
    # Combine padding mask with causal mask to get final decoder mask
    # decoder_padding_masks: [batch, 1, tgt_len] -> [batch, 1, 1, tgt_len]
    # causal_masks: [1, tgt_len, tgt_len] -> [1, 1, tgt_len, tgt_len]
    decoder_padding_masks = decoder_padding_masks.unsqueeze(1)  # [batch, 1, 1, tgt_len]
    causal_masks = causal_masks.unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
    
    # Broadcasting: [batch, 1, 1, tgt_len] & [1, 1, tgt_len, tgt_len] -> [batch, 1, tgt_len, tgt_len]
    decoder_masks = decoder_padding_masks & causal_masks
    
    return {
        "encoder_input": encoder_inputs,
        "decoder_input": decoder_inputs,
        "encoder_mask": encoder_masks,
        "decoder_mask": decoder_masks,
        "label": labels,
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }