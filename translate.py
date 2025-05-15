from pathlib import Path
import argparse
import torch
import sys
import os
import sentencepiece as spm

from model import build_transformer
from dataset import causal_mask
from config import get_config, latest_weights_file_path, get_weights_file_path

def translate(sentence: str, model_path=None, beam_size=1, max_seq_len=None, use_gpu=True):
    """
    Translate a sentence from source language to target language
    Args:
        sentence: Source language sentence
        model_path: Path to the model weights (if None, uses latest)
        beam_size: Beam size for beam search (1 = greedy)
        max_seq_len: Maximum sequence length
        use_gpu: Whether to use GPU for inference
    """
    # Load configuration
    config = get_config()
    
    # Use provided model path or get latest
    if model_path is None:
        model_path = latest_weights_file_path(config)
        if model_path is None:
            print("No model weights found. Please train the model first.")
            return None
    elif not os.path.exists(model_path):
        # Check if it might be a relative path in the weights directory
        weights_dir = f"{config['datasource']}_{config['model_folder']}"
        potential_path = os.path.join(weights_dir, model_path)
        if os.path.exists(potential_path):
            model_path = potential_path
        else:
            print(f"Model weights not found at: {model_path}")
            return None
    
    # Set device
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for translation")
    else:
        device = torch.device("cpu")
        print("Using CPU for translation")
    
    # Set max sequence length
    if max_seq_len is None:
        max_seq_len = config.get('max_len', 500)
    
    # Load tokenizers using SentencePiece
    tokenizer_src = spm.SentencePieceProcessor()
    tokenizer_tgt = spm.SentencePieceProcessor()
    tokenizer_src.load(config['tokenizer_src_path'])
    tokenizer_tgt.load(config['tokenizer_tgt_path'])
    
    # Build model
    model = build_transformer(
        tokenizer_src.vocab_size(),
        tokenizer_tgt.vocab_size(),
        max_seq_len,
        max_seq_len,
        d_model=config.get('d_model', 384),
        N=config.get('num_layers', 4),
        h=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.15),
        d_ff=config.get('d_ff', 1536)
    )
    
    # Load model weights
    print(f"Loading model weights from: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Tokenize input sentence
    with torch.no_grad():
        # Get special token IDs
        sos_idx = tokenizer_src.piece_to_id("<s>")
        eos_idx = tokenizer_src.piece_to_id("</s>")
        pad_idx = tokenizer_src.piece_to_id("<pad>")
        
        # Tokenize and prepare source
        src_tokens = tokenizer_src.encode(sentence)
        
        # Create source tensor with special tokens
        encoder_input = torch.tensor([sos_idx] + src_tokens + [eos_idx], dtype=torch.long).unsqueeze(0).to(device)
        
        # Create source mask
        encoder_mask = (encoder_input != pad_idx).unsqueeze(1).unsqueeze(1).int().to(device)
        
        # Encode source
        encoder_output = model.encode(encoder_input, encoder_mask)
        
        if beam_size == 1:
            # Greedy search
            translation = greedy_decode(model, encoder_output, encoder_mask, tokenizer_tgt, max_seq_len, device)
        else:
            # Beam search
            translation = beam_search(model, encoder_output, encoder_mask, tokenizer_tgt, beam_size, max_seq_len, device)
    
    return translation

def greedy_decode(model, encoder_output, encoder_mask, tokenizer_tgt, max_len, device):
    # Get special token IDs
    sos_idx = tokenizer_tgt.piece_to_id("<s>")
    eos_idx = tokenizer_tgt.piece_to_id("</s>")
    
    # Initialize decoder input with start token
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
    
    # Track generated tokens
    generated_tokens = []
    
    # Generate tokens one by one
    for _ in range(max_len):
        # Create decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        
        # Decode
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        
        # Project and get next token
        prob = model.project(out[:, -1])
        _, next_token = torch.max(prob, dim=1)
        next_token = next_token.item()
        
        # Add token to generated tokens
        generated_tokens.append(next_token)
        
        # Break if end token
        if next_token == eos_idx:
            break
        
        # Update decoder input
        decoder_input = torch.cat([
            decoder_input, 
            torch.tensor([[next_token]], dtype=torch.long).to(device)
        ], dim=1)
    
    # Convert tokens to text
    return tokenizer_tgt.decode(generated_tokens)

def beam_search(model, encoder_output, encoder_mask, tokenizer_tgt, beam_size, max_len, device):
    """
    Beam search implementation for better translation quality.
    
    Args:
        model: The trained transformer model
        encoder_output: Output from the encoder
        encoder_mask: Mask for the encoder output
        tokenizer_tgt: Target language tokenizer
        beam_size: Number of beams to maintain
        max_len: Maximum sequence length
        device: Device to run inference on
        
    Returns:
        The best translation according to beam search
    """
    # Get special token IDs
    sos_idx = tokenizer_tgt.piece_to_id("<s>")
    eos_idx = tokenizer_tgt.piece_to_id("</s>")
    
    # Initialize beam
    # Each item in the beam will be (sequence, score, finished_flag)
    beam = [(torch.tensor([[sos_idx]], device=device), 0.0, False)]
    
    # Track finished sequences and their scores
    finished_sequences = []
    
    # Generate tokens step by step
    for step in range(max_len):
        # Generate candidates for next step
        candidates = []
        
        # Process each sequence in the beam
        for seq, score, finished in beam:
            # Skip already finished sequences
            if finished:
                candidates.append((seq, score, finished))
                continue
                
            # Create decoder mask
            decoder_mask = causal_mask(seq.size(1)).to(device)
            
            # Decode current sequence
            out = model.decode(encoder_output, encoder_mask, seq, decoder_mask)
            
            # Project to vocabulary distribution
            logits = model.project(out[:, -1])
            
            # Get top k tokens and their probabilities
            probs = torch.nn.functional.log_softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, beam_size, dim=-1)
            
            # Create new candidates
            for i in range(beam_size):
                token_idx = topk_indices[0, i].item()
                token_prob = topk_probs[0, i].item()
                
                # Create new sequence
                new_seq = torch.cat([seq, torch.tensor([[token_idx]], device=device)], dim=1)
                
                # Calculate new score (sum of log probabilities)
                new_score = score + token_prob
                
                # Check if this sequence has finished
                is_finished = (token_idx == eos_idx)
                
                # Add to candidates
                if is_finished:
                    # Normalize score by length to avoid bias towards shorter sequences
                    # Use length penalty: (5 + length)^0.6 / (5 + 1)^0.6
                    length_penalty = ((5 + new_seq.size(1)) ** 0.6) / (5 + 1) ** 0.6
                    normalized_score = new_score / length_penalty
                    finished_sequences.append((new_seq, normalized_score, True))
                else:
                    candidates.append((new_seq, new_score, False))
        
        # If we have no active candidates, break
        if len(candidates) == 0:
            break
        
        # Sort candidates by score (descending) and select top beam_size
        # Include finished sequences that might have been added this step
        all_candidates = candidates + finished_sequences
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top beam_size candidates
        beam = all_candidates[:beam_size]
        
        # Extract finished sequences
        finished_sequences = [item for item in beam if item[2]]
        
        # If all beams are finished, break
        if all(finished for _, _, finished in beam):
            break
    
    # Select the best sequence
    if finished_sequences:
        # Choose from finished sequences
        best_seq, _, _ = max(finished_sequences, key=lambda x: x[1])
    else:
        # Choose from current beam
        best_seq, _, _ = max(beam, key=lambda x: x[1])
    
    # Convert to token list without SOS and EOS
    tokens = best_seq.squeeze().tolist()[1:]
    
    # Remove EOS token if present
    if tokens and tokens[-1] == eos_idx:
        tokens = tokens[:-1]
    
    # Convert tokens to text
    return tokenizer_tgt.decode(tokens)

def translate_file(input_file, output_file, model_path=None, beam_size=1, max_seq_len=None, use_gpu=True):
    """Translate all sentences in a file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = f.read().splitlines()
    
    translations = []
    for i, sentence in enumerate(sentences):
        print(f"Translating sentence {i+1}/{len(sentences)}")
        translation = translate(sentence, model_path, beam_size, max_seq_len, use_gpu)
        translations.append(translation)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')
    
    print(f"Translations saved to {output_file}")

def interactive_mode(model_path=None, beam_size=1, max_seq_len=None, use_gpu=True):
    """Interactive translation mode"""
    print("=== Interactive Translation Mode ===")
    print("Enter a sentence to translate (Ctrl+C or type 'exit' to quit)")
    
    try:
        while True:
            print("\nInput:", end=" ")
            sentence = input().strip()
            
            if sentence.lower() == 'exit':
                break
            
            if not sentence:
                continue
            
            translation = translate(sentence, model_path, beam_size, max_seq_len, use_gpu)
            print(f"Translation: {translation}")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")

def main():
    parser = argparse.ArgumentParser(description="Translate text using a trained transformer model")
    
    # Add arguments
    parser.add_argument('--source', type=str, help='Source file to translate')
    parser.add_argument('--output', type=str, help='Output file for translations')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights')
    parser.add_argument('--beam', type=int, default=1, help='Beam size for beam search (default: 1 = greedy)')
    parser.add_argument('--max-length', type=int, default=None, help='Maximum sequence length')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference even if GPU is available')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--text', type=str, help='Translate a single text input')
    
    args = parser.parse_args()
    
    # Choose mode based on arguments
    if args.interactive:
        interactive_mode(args.model, args.beam, args.max_length, not args.cpu)
    elif args.source and args.output:
        translate_file(args.source, args.output, args.model, args.beam, args.max_length, not args.cpu)
    elif args.text:
        translation = translate(args.text, args.model, args.beam, args.max_length, not args.cpu)
        print(f"Translation: {translation}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()