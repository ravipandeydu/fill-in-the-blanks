# Tokenisation & "Fill-in-the-Blank"

This project demonstrates different tokenization algorithms (BPE, WordPiece, and SentencePiece) and implements a fill-in-the-blank prediction task using a large language model.

## Setup

Install the necessary libraries:

```bash
pip install tokenizers transformers sentencepiece
```

## Project Structure

- `tokenise.py`: Implements tokenization using BPE, WordPiece, and SentencePiece
- `predictions.json`: Contains tokenization results and masked token predictions
- `compare.md`: Explains differences between tokenization algorithms
- `README.md`: This file

## Running the Code

### Step 1: Tokenization

Run the tokenization script:

```bash
python tokenise.py
```

This will tokenize the sentence "The cat sat on the mat because it was tired." using three different algorithms and save the results to `predictions.json`.

### Step 2: Mask & Predict

To run the mask prediction part, you'll need to modify the `tokenise.py` file to include the code for loading a language model and predicting masked tokens. The current implementation includes a template for this functionality.

## Assignment Tasks

1. **Tokenisation**: Tokenize the sentence using BPE, WordPiece, and SentencePiece (Unigram)
2. **Report**: For each algorithm, provide token lists, IDs, and counts
3. **Compare**: Explain why the splits differ across algorithms (see `compare.md`)
4. **Mask & Predict**: Replace tokens with mask tokens and predict them using a language model
5. **Evaluate**: Comment on the plausibility of the predictions

## Notes

- The code uses temporary files for training the tokenizers on a single sentence
- For a real-world application, you would train on a much larger corpus
- The predictions in `predictions.json` will be updated when you run the code