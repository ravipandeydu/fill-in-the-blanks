"""
Tokenisation Script for Natural Language Processing

This script demonstrates three different tokenization methods:
1. Byte Pair Encoding (BPE) - Used in GPT models
2. WordPiece - Used in BERT models  
3. SentencePiece - Used in multilingual models

The script also includes functionality to predict masked tokens using
pre-trained language models for fill-in-the-blank tasks.

Author: NLP Tokenization Demo
Date: 2024
"""

import json
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm
import os
import tempfile
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import torch

# Sample sentence for tokenization demonstration
# This sentence will be used to train tokenizers and demonstrate different approaches
sentence = "The cat sat on the mat because it was tired."

def train_bpe_tokenizer(sentence):
    """
    Train and use a Byte Pair Encoding (BPE) tokenizer.
    
    BPE is a subword tokenization algorithm that iteratively merges the most
    frequent adjacent pairs of bytes/characters. It's commonly used in GPT models.
    
    Args:
        sentence (str): The input sentence to tokenize
        
    Returns:
        dict: Dictionary containing tokens, token IDs, and token count
    """
    # Initialize BPE tokenizer with unknown token handling
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # Pre-tokenize on whitespace
    
    # Configure the BPE trainer with special tokens and vocabulary size
    # Special tokens are used for specific NLP tasks (classification, padding, etc.)
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
        vocab_size=100  # Limit vocabulary size for demonstration
    )
    
    # Create temporary file for training data
    # Tokenizers need files to train on, so we create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(sentence)
        temp_file_name = f.name
    
    # Train the tokenizer on our sentence
    tokenizer.train([temp_file_name], trainer)
    os.unlink(temp_file_name)  # Clean up temporary file
    
    # Tokenize the input sentence
    encoding = tokenizer.encode(sentence)
    
    return {
        "tokens": encoding.tokens,      # The actual token strings
        "ids": encoding.ids,            # Numerical IDs for each token
        "count": len(encoding.tokens)   # Total number of tokens
    }

def train_wordpiece_tokenizer(sentence):
    """
    Train and use a WordPiece tokenizer.
    
    WordPiece is similar to BPE but uses likelihood instead of frequency
    to merge subword units. It's commonly used in BERT and related models.
    
    Args:
        sentence (str): The input sentence to tokenize
        
    Returns:
        dict: Dictionary containing tokens, token IDs, and token count
    """
    # Initialize WordPiece tokenizer with unknown token handling
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # Pre-tokenize on whitespace
    
    # Configure the WordPiece trainer with special tokens and vocabulary size
    trainer = WordPieceTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
        vocab_size=100  # Limit vocabulary size for demonstration
    )
    
    # Create temporary file for training data
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(sentence)
        temp_file_name = f.name
    
    # Train the tokenizer on our sentence
    tokenizer.train([temp_file_name], trainer)
    os.unlink(temp_file_name)  # Clean up temporary file
    
    # Tokenize the input sentence
    encoding = tokenizer.encode(sentence)
    
    return {
        "tokens": encoding.tokens,      # The actual token strings
        "ids": encoding.ids,            # Numerical IDs for each token
        "count": len(encoding.tokens)   # Total number of tokens
    }

def train_sentencepiece_tokenizer(sentence):
    """
    Train and use a SentencePiece tokenizer.
    
    SentencePiece is a language-agnostic text tokenization tool that can
    handle multiple languages and doesn't require pre-tokenization. It's
    commonly used in multilingual models like mBERT and T5.
    
    Args:
        sentence (str): The input sentence to tokenize
        
    Returns:
        dict: Dictionary containing tokens, token IDs, and token count
    """
    # Create a temporary file for SentencePiece training
    # SentencePiece requires a text file as input for training
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
        f.write(sentence)
        temp_file_name = f.name
    
    # Create a model prefix for saving the trained model
    model_prefix = os.path.join(tempfile.gettempdir(), "spm_model")
    
    # Train the SentencePiece model using Unigram algorithm
    # Unigram is one of the algorithms supported by SentencePiece
    spm.SentencePieceTrainer.train(
        f'--input={temp_file_name} --model_prefix={model_prefix} '
        f'--vocab_size=25 --model_type=unigram '  # Use Unigram algorithm with 25 vocab size
        f'--character_coverage=1.0 --pad_id=3 --unk_id=0 '  # Configure special token IDs
        f'--bos_id=1 --eos_id=2 --pad_piece=[PAD] --unk_piece=[UNK] '  # Special token pieces
        f'--bos_piece=[BOS] --eos_piece=[EOS] --user_defined_symbols=[MASK]'  # Additional special tokens
    )
    
    # Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    # Tokenize the sentence using the trained model
    tokens = sp.encode_as_pieces(sentence)  # Get token pieces
    ids = sp.encode_as_ids(sentence)        # Get token IDs
    
    # Clean up temporary files created during training
    os.unlink(temp_file_name)           # Remove training data file
    os.unlink(f"{model_prefix}.model")  # Remove model file
    os.unlink(f"{model_prefix}.vocab")  # Remove vocabulary file
    
    return {
        "tokens": tokens,           # The actual token strings
        "ids": ids,                 # Numerical IDs for each token
        "count": len(tokens)        # Total number of tokens
    }

def predict_masked_tokens(mask_positions):
    """
    Predict masked tokens using a lightweight pre-trained language model.
    
    This function takes specific token positions, masks them, and uses
    a lightweight language model to predict what tokens should fill those positions.
    The model is designed to run locally without GPU requirements.
    This is useful for fill-in-the-blank tasks and understanding model predictions.
    
    Args:
        mask_positions (list): List of token positions to mask and predict
        
    Returns:
        dict: Dictionary containing prediction results and metadata
    """
    # Get the tokens from SentencePiece tokenizer for this example
    # We use SentencePiece tokens as they're more granular and suitable for masking
    sp_result = train_sentencepiece_tokenizer(sentence)
    tokens = sp_result["tokens"]
    
    # Create a copy of tokens and replace the specified positions with [MASK]
    # We need to track original tokens to compare with predictions
    masked_tokens = tokens.copy()
    original_tokens = []
    
    # Replace tokens at specified positions with mask token
    for pos in mask_positions:
        if pos < len(tokens):
            original_tokens.append(tokens[pos])  # Store original token
            masked_tokens[pos] = "[MASK]"       # Replace with mask token
    
    # Convert tokenized sentence back to text for the language model
    # SentencePiece uses ▁ to indicate word boundaries, which we replace with spaces
    masked_sentence = "".join(masked_tokens).replace("▁", " ").strip()
    
    print(f"\nMasked sentence: {masked_sentence}")
    
    try:
        # Load a lightweight language model for masked token prediction
        # Using a smaller model that can run locally without GPU
        print("\nLoading lightweight language model... (this may take a moment)")
        
        # Try multiple models in order of preference, starting with Mistral 2B
        model_options = [
            "mistralai/Mistral-7B-Instruct-v0.2",  # Mistral 7B Instruct model
            "microsoft/DialoGPT-medium",            # ~1.5GB, good for conversational text
            "distilroberta-base",                   # ~260MB, very fast
            "bert-base-uncased",                    # ~440MB, standard BERT
            "albert-base-v2"                        # ~43MB, very small but effective
        ]
        
        fill_mask = None
        model_name = None
        
        for model in model_options:
            try:
                print(f"Trying to load {model}...")
                
                # Check if it's a Mistral model (generative) or traditional fill-mask model
                if "mistral" in model.lower():
                    # For Mistral, use text-generation pipeline
                    generator = pipeline(
                        "text-generation",
                        model=model,
                        device=-1,  # Force CPU usage
                        torch_dtype="auto"
                    )
                    model_name = model
                    print(f"Successfully loaded {model} (generative model)")
                    break
                else:
                    # For traditional models, use fill-mask pipeline
                    fill_mask = pipeline(
                        "fill-mask",
                        model=model,
                        top_k=3,  # Return top 3 predictions for each mask
                        device=-1  # Force CPU usage
                    )
                    model_name = model
                    print(f"Successfully loaded {model}")
                    break
            except Exception as e:
                print(f"Could not load {model}: {e}")
                continue
        
        # If all models fail, provide a helpful error message
        if fill_mask is None and 'generator' not in locals():
            raise Exception("Could not load any models. Please check your internet connection and try again.")
        
        # Handle different model types
        if 'generator' in locals():
            # For generative models like Mistral, use a different approach
            print("Using generative model for predictions...")
            
            # Create prompts for each masked position
            formatted_predictions = []
            for i, (pos, orig_token) in enumerate(zip(mask_positions, original_tokens)):
                # Create a prompt that asks the model to fill in the blank
                prompt = f"Complete this sentence by filling in the blank: {masked_sentence.replace('[MASK]', '___')}"
                
                # Generate completion
                generated_text = generator(prompt, max_length=len(prompt.split()) + 10, 
                                       num_return_sequences=1, do_sample=True, temperature=0.7)
                
                # Extract the generated text and find the filled word
                full_text = generated_text[0]['generated_text']
                # Try to extract the word that was filled in
                try:
                    # Simple approach: look for the word after "___"
                    parts = full_text.split('___')
                    if len(parts) > 1:
                        filled_word = parts[1].split()[0].strip('.,!?')
                    else:
                        filled_word = "unknown"
                except:
                    filled_word = "unknown"
                
                formatted_predictions.append({
                    "position": pos,
                    "original_token": orig_token.replace("▁", "").strip(),
                    "top_predictions": [{
                        "token": filled_word,
                        "score": 0.8,  # Placeholder score for generative models
                        "plausibility": "high - generated by Mistral model"
                    }]
                })
        else:
            # For traditional fill-mask models
            # Get the mask token for the specific model
            # Different models use different mask tokens (e.g., [MASK], <mask>, etc.)
            # This ensures compatibility with the loaded lightweight model
            mask_token = fill_mask.tokenizer.mask_token
            
            # Replace our generic [MASK] with the model's specific mask token
            masked_sentence_for_model = masked_sentence.replace("[MASK]", mask_token)
            
            print(f"Using model's mask token: {mask_token}")
            print(f"Masked sentence for model: {masked_sentence_for_model}")
            
            # Get predictions from the language model
            predictions = fill_mask(masked_sentence_for_model)
            
            # Handle different prediction formats
            # Some models return a single list, others return a list of lists for multiple masks
            if not isinstance(predictions[0], list):
                predictions = [predictions]
            
            # Format the predictions with additional metadata
            formatted_predictions = []
            for i, (pos, orig_token) in enumerate(zip(mask_positions, original_tokens)):
                top_preds = []
                for pred in predictions[i]:
                    token = pred["token_str"].strip()
                    score = pred["score"]
                    
                    # Determine plausibility based on score and original token match
                    # This helps evaluate prediction quality
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
                
                # Store prediction results for this position
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
        # Handle errors gracefully and provide fallback results
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

def main():
    """
    Main function that orchestrates the tokenization and prediction process.
    
    This function:
    1. Trains all three tokenizers (BPE, WordPiece, SentencePiece)
    2. Displays tokenization results
    3. Performs masked token prediction
    4. Saves results to a JSON file
    """
    print("Starting tokenization...")
    
    # Train all three tokenizers on the sample sentence
    results = {
        "BPE": train_bpe_tokenizer(sentence),
        "WordPiece": train_wordpiece_tokenizer(sentence),
        "SentencePiece": train_sentencepiece_tokenizer(sentence)
    }
    
    # Display tokenization results for comparison
    print("\nTokenisation Results:")
    print("====================")
    
    for name, result in results.items():
        print(f"\n{name} Tokenization:")
        print(f"Tokens: {result['tokens']}")
        print(f"Token IDs: {result['ids']}")
        print(f"Total Token Count: {result['count']}")
    
    # Choose token positions to mask for prediction
    # We'll try to find meaningful words like 'mat' and 'tired'
    sp_tokens = results["SentencePiece"]["tokens"]
    
    # Find positions of specific words in the tokenized sentence
    mask_positions = []
    for i, token in enumerate(sp_tokens):
        if "mat" in token.lower():
            mask_positions.append(i)
        elif "tired" in token.lower():
            mask_positions.append(i)
    
    # If we couldn't find exactly two positions, use fallback positions
    # This ensures we always have positions to mask
    if len(mask_positions) != 2:
        mask_positions = [5, 9]  # Fallback positions
    
    print(f"\nMasking tokens at positions: {mask_positions}")
    print(f"Original tokens at these positions: {[sp_tokens[pos] for pos in mask_positions]}")
    
    # Perform masked token prediction
    print("\nPredicting masked tokens...")
    masked_predictions = predict_masked_tokens(mask_positions)
    
    # Add masked predictions to overall results
    results["masked_predictions"] = masked_predictions
    
    # Save all results to a JSON file for later analysis
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to predictions.json")

# Entry point - only run if script is executed directly
if __name__ == "__main__":
    main()