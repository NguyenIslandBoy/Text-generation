# Agatha Christie Text Generator

A word-level recurrent neural network that generates text in the style of Agatha Christie, trained on a single Project Gutenberg ebook (~68,000 tokens).

## Architecture

| Component | Detail |
|-----------|--------|
| Model | 2-layer Stacked GRU |
| Tokenisation | Word-level (regex: `\w+\|[^\w\s]`) |
| Vocabulary | 3,295 tokens (MIN_FREQ=2) |
| Embedding | 256 dimensions |
| GRU units | 512 × 2 layers |
| Dropout | 0.2 |
| Context window | 50 words |
| Sampling | Top-k (k=40) + Top-p (p=0.9) + temperature scaling |
| Final metrics | Loss ≈ 0.4, Accuracy ≈ 90% |

## Files

```
├── Coursework_Language_Model.ipynb   # Full training and inference code
├── language model.keras              # Trained model weights (62 MB)
├── vocab.json                        # Vocabulary mappings (token ↔ ID)
└── README.md
```

## Quick Start (Testing Only)

1. Open `Coursework_Language_Model.ipynb` in Google Colab
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Run the following cells in order:

| Step | Cell | Description |
|------|------|-------------|
| 1 | Imports | `import tensorflow as tf ...` |
| 2 | Mount Drive | `drive.mount(...)` |
| 3 | Load Model | Load `language model.keras` + `vocab.json` |
| 4 | TextGenerator class | Defines the generation logic |
| 5 | Demo cell (last cell) | Interactive prompt — enter text, get output |

No retraining required. The saved model weights and vocabulary are sufficient for inference.

## Full Training

Run all cells top to bottom. Requires:
- Google Colab with T4 GPU (or better)
- `61262-0.txt` (Project Gutenberg ebook) in Google Drive
- Training time: ~15 minutes for 150 epochs

## Example Outputs

**Prompt:** "Poirot said"
> Poirot said that he looked at me and grunted in his ear. "Pardon, but could you give you entirely out of Monsieur Poirot." He looked at me and grunted-faced chair — a tall, handsome woman...

**Prompt:** "She looked at"
> She looked at Ealing, and then the world had placed certain out of the window. They arrived a certain job at Ealing and the same quite...

## Key Design Decisions

- **Word-level over character-level:** Produces real words and more coherent phrases, at the cost of a larger vocabulary and slower convergence.
- **MIN_FREQ=2:** Tested thresholds 2–5. Higher values (e.g., 5) reduced vocab to 1,385 but caused 12% UNK rate, flooding output with unknown tokens. MIN_FREQ=2 gives 4.7% UNK rate — the best available tradeoff.
- **UNK suppression at inference:** The `<UNK>` token logit is set to −10⁹ before sampling, preventing unknown tokens in generated text.
- **Top-k + Top-p sampling:** Avoids the repetitiveness of greedy decoding while preventing incoherent outputs from unrestricted sampling.

## Tools

Python 3.12 · TensorFlow 2.16 · Google Colab (T4 GPU)
