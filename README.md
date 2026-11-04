# Metali-C RUbrique 

TextCNN for age-based classification of Russian children's fiction, written from scratch in Ruby.

Pure Ruby TextCNN (Numo::NArray) replicates my [master's thesis](https://github.com/mikske/thesis_rusage), where the model was written in PyTorch.

## Data 

- Corpus: [RusAge](https://www.kaggle.com/datasets/oldaandozerskaya/fiction-corpus-for-agebased-text-classification), with 2 classes: ages 5–6 and 7–12
- **2,380 documents → 31,413 chunks** (up to 700 tokens each), split by document so that no text appears in more than one set:

| Split | Docs | Chunks | Ages 5–6 / 7–12 (chunks) | `<unk>` rate |
|---|---|---|---|---|
| train | 1,666 | 21,851 | 7,520 / 14,331 | 9.8% |
| val | 357 | 5,116 | 1,639 / 3,477 | 10.5% |
| test | 357 | 4,446 | 1,501 / 2,945 | 10.6% |

- At the document level the two classes are almost balanced (1,228 vs 1,152), but at the chunk level the ratio is about 1:2, because books for older children are longer.
- Embeddings: static fastText vectors (300 dimensions, 30k vocabulary), exported from PyTorch to raw float32.
- The data is not included in the repository.

## Verified against the PyTorch version
- Embedding matrix: the norms match `torch` to 4 decimal places
- Tokenizer: identical token counts on all 31,413 chunks

## Roadmap
        <-------- We're here currently
- [ ] Pure Ruby TextCNN (Numo::NArray) 
- [ ] C kernel for convolution
- [ ] Metal GPU backend
- [ ] Benchmarks vs PyTorch

## Why?
Why not