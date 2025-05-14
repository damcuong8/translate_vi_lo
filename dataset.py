import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import sentencepiece as spm

class BilingualDataset(Dataset):

    def __init__(self, src_file, tgt_file, tokenizer_src, tokenizer_tgt, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        
        # Load the source and target texts
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_texts = f.read().splitlines()
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_texts = f.read().splitlines()
            
        # Ensure src and tgt have the same number of sentences
        assert len(self.src_texts) == len(self.tgt_texts), "Source and target files must have the same number of lines"

        # Get special token IDs
        self.sos_token = torch.tensor([tokenizer_tgt.piece_to_id("<s>")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.piece_to_id("</s>")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.piece_to_id("<pad>")], dtype=torch.int64)

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Transform the text into tokens using SentencePiece
        enc_input_tokens = self.tokenizer_src.encode(src_text)
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text)

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0