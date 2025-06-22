# Comparison of Tokenization Algorithms

## Token Splits Across Algorithms

### BPE (Byte Pair Encoding)
- BPE works by iteratively merging the most frequent pairs of bytes or characters in the training data.
- It starts with individual characters and builds up a vocabulary of subword units.
- BPE tends to create a balance between character-level and word-level tokens.
- For rare words, BPE will often split them into smaller subword units.

### WordPiece
- WordPiece is similar to BPE but uses a different merging criterion based on likelihood rather than frequency.
- It uses a greedy longest-match-first approach when tokenizing.
- WordPiece often adds '##' prefix to subword units that don't start a word.
- It's commonly used in BERT and other transformer models from Google.

### SentencePiece (Unigram)
- SentencePiece treats the input as a raw Unicode sequence without pre-tokenization.
- The Unigram model starts with a large vocabulary and iteratively removes tokens to maximize the likelihood of the training data.
- SentencePiece typically adds '▁' (U+2581) at the beginning of tokens that start a word.
- This approach allows for completely language-agnostic tokenization.

## Why the Splits Differ

The tokenization results differ across these algorithms due to their fundamental approaches:

1. **Training Methodology**: BPE builds up from characters by merging, WordPiece uses likelihood-based merging, and Unigram starts with a large vocabulary and prunes it down.

2. **Word Boundaries**: SentencePiece preserves word boundary information with the '▁' prefix, while BPE and WordPiece typically rely on pre-tokenization.

3. **Subword Formation**: Each algorithm has different criteria for forming subwords - BPE uses frequency, WordPiece uses likelihood, and Unigram uses a probabilistic approach.

4. **Handling of Rare Words**: The algorithms differ in how they handle uncommon words, with some breaking them into smaller units than others.

These differences result in varying token counts and boundaries, which can impact model performance for different languages and tasks.