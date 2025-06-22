# Tokenisation using BPE, WordPiece, and SentencePiece

import json
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm
import os
import tempfile
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

# The sentence to tokenize
sentence = "The cat sat on the mat because it was tired."

def train_bpe_tokenizer(sentence):
    # Create a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Create a trainer
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=100)
    
    # Train the tokenizer
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(sentence)
        temp_file_name = f.name
    
    tokenizer.train([temp_file_name], trainer)
    os.unlink(temp_file_name)  # Delete the temporary file
    
    # Tokenize the sentence
    encoding = tokenizer.encode(sentence)
    
    return {
        "tokens": encoding.tokens,
        "ids": encoding.ids,
        "count": len(encoding.tokens)
    }

def train_wordpiece_tokenizer(sentence):
    # Create a WordPiece tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Create a trainer
    trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=100)
    
    # Train the tokenizer
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(sentence)
        temp_file_name = f.name
    
    tokenizer.train([temp_file_name], trainer)
    os.unlink(temp_file_name)  # Delete the temporary file
    
    # Tokenize the sentence
    encoding = tokenizer.encode(sentence)
    
    return {
        "tokens": encoding.tokens,
        "ids": encoding.ids,
        "count": len(encoding.tokens)
    }

def train_sentencepiece_tokenizer(sentence):
    # Create a temporary file for SentencePiece training
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
        f.write(sentence)
        temp_file_name = f.name
    
    # Create a model prefix
    model_prefix = os.path.join(tempfile.gettempdir(), "spm_model")
    
    # Train the SentencePiece model (Unigram)
    spm.SentencePieceTrainer.train(
        f'--input={temp_file_name} --model_prefix={model_prefix} '
        f'--vocab_size=100 --model_type=unigram '
        f'--character_coverage=1.0 --pad_id=3 --unk_id=0 '
        f'--bos_id=1 --eos_id=2 --pad_piece=[PAD] --unk_piece=[UNK] '
        f'--bos_piece=[BOS] --eos_piece=[EOS] --user_defined_symbols=[MASK]'
    )
    
    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    # Tokenize the sentence
    tokens = sp.encode_as_pieces(sentence)
    ids = sp.encode_as_ids(sentence)
    
    # Clean up temporary files
    os.unlink(temp_file_name)
    os.unlink(f"{model_prefix}.model")
    os.unlink(f"{model_prefix}.vocab")
    
    return {
        "tokens": tokens,
        "ids": ids,
        "count": len(tokens)
    }

def predict_masked_tokens(mask_positions):
    # Get the tokens from one of the tokenizers (using SentencePiece for this example)
    sp_result = train_sentencepiece_tokenizer(sentence)
    tokens = sp_result["tokens"]
    
    # Create a copy of tokens and replace the specified positions with [MASK]
    masked_tokens = tokens.copy()
    original_tokens = []
    
    for pos in mask_positions:
        if pos < len(tokens):
            original_tokens.append(tokens[pos])
            masked_tokens[pos] = "[MASK]"
    
    # Convert back to a sentence
    masked_sentence = "".join(masked_tokens).replace("▁", " ").strip()
    
    print(f"\nMasked sentence: {masked_sentence}")
    
    try:
        # Load a language model for masked token prediction
        # Note: This will download a large model, which might take time
        print("\nLoading language model... (this may take a while)")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # You can change this to any model that supports fill-mask
        
        # For demonstration purposes, we'll use a smaller model if available
        try:
            # Try to use a smaller model first for faster execution
            fill_mask = pipeline(
                "fill-mask",
                model="distilroberta-base",  # Smaller model for demonstration
                top_k=3
            )
            model_name = "distilroberta-base"  # Update the model name
        except Exception as e:
            print(f"Could not load smaller model: {e}")
            print("Falling back to larger model...")
            # Fall back to the larger model
            fill_mask = pipeline(
                "fill-mask",
                model=model_name,
                top_k=3
            )
        
        # Get the mask token for the model
        mask_token = fill_mask.tokenizer.mask_token
        
        # Replace [MASK] with the model's specific mask token
        masked_sentence_for_model = masked_sentence.replace("[MASK]", mask_token)
        
        print(f"Using model's mask token: {mask_token}")
        print(f"Masked sentence for model: {masked_sentence_for_model}")
        
        # Get predictions
        predictions = fill_mask(masked_sentence_for_model)
        
        # If we have multiple masks, the result will be a list of lists
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        # Format the predictions
        formatted_predictions = []
        for i, (pos, orig_token) in enumerate(zip(mask_positions, original_tokens)):
            top_preds = []
            for pred in predictions[i]:
                token = pred["token_str"].strip()
                score = pred["score"]
                
                # Determine plausibility
                if token == orig_token.replace("▁", "").strip():
                    plausibility = "high - matches original context"
                elif score > 0.2:
                    plausibility = "high - semantically appropriate"
                elif score > 0.05:
                    plausibility = "medium - somewhat appropriate"
                else:
                    plausibility = "low - not very appropriate"
                
                top_preds.append({
                    "token": token,
                    "score": score,
                    "plausibility": plausibility
                })
            
            formatted_predictions.append({
                "position": pos,
                "original_token": orig_token.replace("▁", "").strip(),
                "top_predictions": top_preds
            })
        
        return {
            "model": model_name,
            "masked_sentence": masked_sentence,
            "mask_positions": mask_positions,
            "predictions": formatted_predictions
        }
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return a placeholder if prediction fails
        return {
            "model": "Model loading failed",
            "masked_sentence": masked_sentence,
            "mask_positions": mask_positions,
            "error": str(e),
            "predictions": [{
                "position": pos,
                "original_token": token.replace("▁", "").strip(),
                "top_predictions": [{
                    "token": "[PREDICTION FAILED]",
                    "score": 0.0,
                    "plausibility": "N/A - model loading failed"
                }]
            } for pos, token in zip(mask_positions, original_tokens)]
        }

# Run all tokenizers
def main():
    print("Starting tokenization...")
    results = {
        "BPE": train_bpe_tokenizer(sentence),
        "WordPiece": train_wordpiece_tokenizer(sentence),
        "SentencePiece": train_sentencepiece_tokenizer(sentence)
    }
    
    # Print results
    print("\nTokenisation Results:")
    print("====================")
    
    for name, result in results.items():
        print(f"\n{name} Tokenization:")
        print(f"Tokens: {result['tokens']}")
        print(f"Token IDs: {result['ids']}")
        print(f"Total Token Count: {result['count']}")
    
    # Choose two positions to mask (e.g., 'mat' and 'tired')
    # Find the positions in the SentencePiece tokenization
    sp_tokens = results["SentencePiece"]["tokens"]
    
    # Find positions of 'mat' and 'tired' or similar tokens
    mask_positions = []
    for i, token in enumerate(sp_tokens):
        if "mat" in token.lower():
            mask_positions.append(i)
        elif "tired" in token.lower():
            mask_positions.append(i)
    
    # If we couldn't find exactly two positions, choose positions 5 and 9 as fallback
    if len(mask_positions) != 2:
        mask_positions = [5, 9]  # Assuming these are reasonable positions
    
    print(f"\nMasking tokens at positions: {mask_positions}")
    print(f"Original tokens at these positions: {[sp_tokens[pos] for pos in mask_positions]}")
    
    # Predict masked tokens
    print("\nPredicting masked tokens...")
    masked_predictions = predict_masked_tokens(mask_positions)
    
    # Add masked predictions to results
    results["masked_predictions"] = masked_predictions
    
    # Save results to JSON file
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to predictions.json")

if __name__ == "__main__":
    main()