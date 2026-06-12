# Financial Sentiment Analysis — DistilBERT Fine-tuning

Fine-tuned **DistilBERT** (HuggingFace Transformers + PyTorch) on the Financial PhraseBank dataset for 3-class sentiment classification of financial news sentences.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **84%** |
| Macro F1 | **0.83** |
| Negative F1 | 0.83 |
| Neutral F1 | 0.87 |
| Positive F1 | 0.78 |

## Training Loss

![Training Loss](<img width="790" height="390" alt="download" src="https://github.com/user-attachments/assets/64778580-707e-4a1d-a611-a95cda379b95" />
)

Loss converged from **0.5695 → 0.2696 → 0.1605** over 3 epochs.

## Confusion Matrix

![Confusion Matrix](<img width="652" height="490" alt="download (1)" src="https://github.com/user-attachments/assets/f3cb7ad4-ab7b-4123-9e2d-8547579f77c4" />
)

## Dataset

[Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) — 4,846 financial news sentences labeled Positive / Negative / Neutral.

| Class | Count |
|-------|-------|
| Neutral | 2,879 |
| Positive | 1,363 |
| Negative | 604 |

## Model Architecture

- **Base model**: `distilbert-base-uncased`
- **Task**: Sequence Classification (3 labels)
- **Optimizer**: AdamW (lr=2e-5)
- **Scheduler**: Linear with no warmup
- **Epochs**: 3
- **Batch size**: 16
- **Max token length**: 128

## Key Decisions

- Used **DistilBERT** over full BERT — 40% smaller, 60% faster, retains 97% of BERT's performance
- **Stratified train/test split** (80/20) to preserve class imbalance ratios
- Fine-tuned all layers (not frozen) for domain adaptation to financial language

## Stack

Python · PyTorch · HuggingFace Transformers · DistilBERT · scikit-learn · pandas · seaborn · matplotlib
