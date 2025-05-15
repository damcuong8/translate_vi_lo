import argparse
import torch
import sentencepiece as spm
from tqdm import tqdm
import time
import os
from pathlib import Path

from model import build_transformer
from config import get_config, latest_weights_file_path
from translate import translate, beam_search
from dataset import causal_mask

# Import metrics
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

def load_tokenizers(config):
    """Load the source and target tokenizers"""
    tokenizer_src = spm.SentencePieceProcessor()
    tokenizer_tgt = spm.SentencePieceProcessor()
    tokenizer_src.load(config['tokenizer_src_path'])
    tokenizer_tgt.load(config['tokenizer_tgt_path'])
    return tokenizer_src, tokenizer_tgt

def load_model(config, model_path=None, device='cuda'):
    """Load the transformer model"""
    # Determine which model weights to use
    if model_path is None:
        model_path = latest_weights_file_path(config)
        if model_path is None:
            raise ValueError("No model weights found. Please train the model first.")
    elif not os.path.exists(model_path):
        # Check if it might be a relative path in the weights directory
        weights_dir = f"{config['datasource']}_{config['model_folder']}"
        potential_path = os.path.join(weights_dir, model_path)
        if os.path.exists(potential_path):
            model_path = potential_path
        else:
            raise ValueError(f"Model weights not found at: {model_path}")
    
    # Load tokenizers to get vocabulary sizes
    tokenizer_src, tokenizer_tgt = load_tokenizers(config)
    
    # Build model
    print(f"Building model...")
    model = build_transformer(
        tokenizer_src.vocab_size(),
        tokenizer_tgt.vocab_size(),
        config.get('max_len', 500),
        config.get('max_len', 500),
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
    
    return model, tokenizer_src, tokenizer_tgt

def batch_translate(model, sentences, tokenizer_src, tokenizer_tgt, device, 
                   batch_size=8, beam_size=1, max_len=None):
    """
    Translate a batch of sentences for faster processing
    
    Args:
        model: The trained transformer model
        sentences: List of sentences to translate
        tokenizer_src: Source language tokenizer
        tokenizer_tgt: Target language tokenizer
        device: Device to run inference on
        batch_size: Batch size for processing
        beam_size: Beam size for beam search (1 = greedy)
        max_len: Maximum sequence length
        
    Returns:
        List of translated sentences
    """
    # Set default max_len
    if max_len is None:
        max_len = 500
    
    # Special token IDs
    sos_idx = tokenizer_src.piece_to_id("<s>")
    eos_idx = tokenizer_src.piece_to_id("</s>")
    pad_idx = tokenizer_src.piece_to_id("<pad>")
    
    translations = []
    
    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        batch_size_actual = len(batch_sentences)
        
        # Encode all sentences in the batch
        with torch.no_grad():
            # Create a list of tokenized sentences
            batch_tokens = [tokenizer_src.encode(sent) for sent in batch_sentences]
            
            # Find max length in this batch
            max_len_batch = max(len(tokens) for tokens in batch_tokens) + 2  # +2 for SOS and EOS
            
            # Create input tensor with padding
            encoder_inputs = torch.full((batch_size_actual, max_len_batch), 
                                        pad_idx, dtype=torch.long, device=device)
            
            # Fill in the actual tokens
            for j, tokens in enumerate(batch_tokens):
                # Add SOS and EOS tokens
                encoder_inputs[j, 0] = sos_idx
                encoder_inputs[j, 1:len(tokens)+1] = torch.tensor(tokens, dtype=torch.long, device=device)
                encoder_inputs[j, len(tokens)+1] = eos_idx
            
            # Create encoder masks (1 for tokens, 0 for padding)
            encoder_masks = (encoder_inputs != pad_idx).unsqueeze(1).unsqueeze(1).int()
            
            # Encode all sentences in batch
            encoder_outputs = model.encode(encoder_inputs, encoder_masks)
            
            # Translate each sentence in the batch
            batch_translations = []
            for j in range(batch_size_actual):
                # Extract this sentence's encoder output and mask
                encoder_output = encoder_outputs[j:j+1]  # Keep batch dimension
                encoder_mask = encoder_masks[j:j+1]      # Keep batch dimension
                
                if beam_size == 1:
                    # Greedy search
                    # Initialize decoder input
                    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
                    
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
                            torch.tensor([[next_token]], dtype=torch.long, device=device)
                        ], dim=1)
                    
                    # Convert tokens to text
                    translation = tokenizer_tgt.decode(generated_tokens)
                else:
                    # Use beam search for better quality
                    # Initialize beam
                    beam = [(torch.tensor([[sos_idx]], device=device), 0.0, False)]
                    finished_sequences = []
                    
                    # Generate tokens step by step
                    for step in range(max_len):
                        candidates = []
                        
                        for seq, score, finished in beam:
                            if finished:
                                candidates.append((seq, score, finished))
                                continue
                            
                            decoder_mask = causal_mask(seq.size(1)).to(device)
                            out = model.decode(encoder_output, encoder_mask, seq, decoder_mask)
                            logits = model.project(out[:, -1])
                            
                            probs = torch.nn.functional.log_softmax(logits, dim=-1)
                            topk_probs, topk_indices = torch.topk(probs, beam_size, dim=-1)
                            
                            for k in range(beam_size):
                                token_idx = topk_indices[0, k].item()
                                token_prob = topk_probs[0, k].item()
                                
                                new_seq = torch.cat([seq, torch.tensor([[token_idx]], device=device)], dim=1)
                                new_score = score + token_prob
                                is_finished = (token_idx == eos_idx)
                                
                                if is_finished:
                                    length_penalty = ((5 + new_seq.size(1)) ** 0.6) / (5 + 1) ** 0.6
                                    normalized_score = new_score / length_penalty
                                    finished_sequences.append((new_seq, normalized_score, True))
                                else:
                                    candidates.append((new_seq, new_score, False))
                        
                        if len(candidates) == 0:
                            break
                        
                        all_candidates = candidates + finished_sequences
                        all_candidates.sort(key=lambda x: x[1], reverse=True)
                        beam = all_candidates[:beam_size]
                        
                        finished_sequences = [item for item in beam if item[2]]
                        
                        if all(finished for _, _, finished in beam):
                            break
                    
                    # Select the best sequence
                    if finished_sequences:
                        best_seq, _, _ = max(finished_sequences, key=lambda x: x[1])
                    else:
                        best_seq, _, _ = max(beam, key=lambda x: x[1])
                    
                    # Convert to token list without SOS and EOS
                    tokens = best_seq.squeeze().tolist()[1:]
                    
                    # Remove EOS token if present
                    if tokens and tokens[-1] == eos_idx:
                        tokens = tokens[:-1]
                    
                    # Convert tokens to text
                    translation = tokenizer_tgt.decode(tokens)
                
                batch_translations.append(translation)
            
            translations.extend(batch_translations)
    
    return translations

def evaluate(test_file, reference_file, model_path=None, beam_size=1, batch_size=32, use_gpu=True):
    """
    Evaluate the translation model on a test set
    
    Args:
        test_file: Path to file with source sentences
        reference_file: Path to file with reference translations
        model_path: Path to model weights
        beam_size: Beam size for beam search
        batch_size: Batch size for processing
        use_gpu: Whether to use GPU for inference
        
    Returns:
        Dict of metrics: BLEU, WER, CER
    """
    # Load configuration
    config = get_config()
    
    # Set device
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for evaluation")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation")
    
    # Load model
    model, tokenizer_src, tokenizer_tgt = load_model(config, model_path, device)
    
    # Read test and reference files
    with open(test_file, 'r', encoding='utf-8') as f:
        test_sentences = f.read().splitlines()
    
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_sentences = f.read().splitlines()
    
    if len(test_sentences) != len(reference_sentences):
        raise ValueError("Test and reference files must have the same number of lines")
    
    print(f"Evaluating on {len(test_sentences)} sentences...")
    
    # Translate test sentences
    start_time = time.time()
    translated_sentences = batch_translate(
        model, test_sentences, tokenizer_src, tokenizer_tgt, device,
        batch_size=batch_size, beam_size=beam_size, max_len=config.get('max_len', 500)
    )
    end_time = time.time()
    
    # Compute metrics
    metrics = {}
    
    # Compute BLEU score
    try:
        bleu_metric = BLEUScore()
        
        # Tokenize predictions and references
        tokenized_predictions = [sent.split() for sent in translated_sentences]
        tokenized_references = [[sent.split()] for sent in reference_sentences]  # List of lists for multiple references
        
        bleu = bleu_metric(tokenized_predictions, tokenized_references)
        metrics['bleu'] = bleu.item()
    except Exception as e:
        print(f"Error calculating BLEU: {str(e)}")
        metrics['bleu'] = 0.0
    
    # Compute WER
    try:
        wer_metric = WordErrorRate()
        wer = wer_metric(translated_sentences, reference_sentences)
        metrics['wer'] = wer.item()
    except Exception as e:
        print(f"Error calculating WER: {str(e)}")
        metrics['wer'] = 1.0  # Worst case
    
    # Compute CER
    try:
        cer_metric = CharErrorRate()
        cer = cer_metric(translated_sentences, reference_sentences)
        metrics['cer'] = cer.item()
    except Exception as e:
        print(f"Error calculating CER: {str(e)}")
        metrics['cer'] = 1.0  # Worst case
    
    # Print results
    print("\nEvaluation results:")
    print(f"BLEU score: {metrics['bleu']:.4f}")
    print(f"Word Error Rate: {metrics['wer']:.4f}")
    print(f"Character Error Rate: {metrics['cer']:.4f}")
    print(f"Evaluation time: {end_time - start_time:.2f} seconds")
    print(f"Average time per sentence: {(end_time - start_time) / len(test_sentences):.4f} seconds")
    
    # Save evaluation results including examples
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"eval_results_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation results:\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Beam size: {beam_size}\n")
        f.write(f"Test file: {test_file}\n")
        f.write(f"Reference file: {reference_file}\n\n")
        
        f.write(f"BLEU score: {metrics['bleu']:.4f}\n")
        f.write(f"Word Error Rate: {metrics['wer']:.4f}\n")
        f.write(f"Character Error Rate: {metrics['cer']:.4f}\n")
        f.write(f"Evaluation time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Average time per sentence: {(end_time - start_time) / len(test_sentences):.4f} seconds\n\n")
        
        f.write("Examples (first 10):\n")
        for i in range(min(10, len(test_sentences))):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Source: {test_sentences[i]}\n")
            f.write(f"Reference: {reference_sentences[i]}\n")
            f.write(f"Prediction: {translated_sentences[i]}\n")
    
    print(f"Evaluation results saved to {output_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate translation model")
    
    parser.add_argument('--test_file', type=str, required=True, 
                        help='Path to test file with source sentences')
    parser.add_argument('--reference', type=str, required=True, 
                        help='Path to reference file with target translations')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to model weights (default: latest)')
    parser.add_argument('--beam', type=int, default=5, 
                        help='Beam size for beam search (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--cpu', action='store_true', 
                        help='Force CPU inference even if GPU is available')
    
    args = parser.parse_args()
    
    evaluate(
        args.test_file,
        args.reference,
        args.model,
        args.beam,
        args.batch_size,
        not args.cpu
    )

if __name__ == "__main__":
    main() 