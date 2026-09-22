# Optimism-and-Pessimism-Detection-in-Tweets

## Overview
 
Sentiment classifiers capture polarity (positive/negative) but often miss the more nuanced, future-oriented psychological stances of optimism and pessimism. This project tackles tweet-level optimism vs. pessimism detection as a binary classification task, addressing a key practical challenge: the source dataset (SemEval-2018 Task 1 E-c) is multi-label, and over 60% of tweets have no explicit optimism or pessimism annotation.
 
We propose a label engineering approach that derives soft optimism scores from the co-occurrence of nine basic emotions, then compare three label assignment strategies and five model families under two preprocessing regimes, evaluating both in-domain performance and cross-dataset generalization.
 
## Datasets
 
- **SemEval-2018 Task 1 E-c** — primary training/dev dataset, multi-label emotion annotations.
- **Dell Sentiment & Emotions Labelled Tweets** (~25,000 brand-mention tweets, 2022) — used for cross-dataset generalization testing, with labels engineered from sentiment scores and reused SemEval emotion weights.
## Method
 
### Label Engineering
For each of the nine basic emotions, we compute a weight reflecting how strongly it co-occurs with optimism vs. pessimism, based on empirical conditional probabilities from the SemEval training set. Tweets lacking explicit labels are then assigned an aggregate optimism score from the emotions present in the text.
 
Three label assignment strategies are compared:
- **A1 (Zero Threshold):** classify by the sign of the score; maximizes dataset size.
- **A2 (Percentile Filtering):** drop the middle 33% of scores as a neutral zone, keeping only high-confidence examples.
- **B (Strict):** keep only tweets where optimism and pessimism explicitly disagree; cleanest but smallest dataset.
### Preprocessing
Two regimes are compared: **minimal** preprocessing (Unicode normalization, URL removal, whitespace cleanup, casing/punctuation/emojis preserved) and **model-specific** preprocessing tailored to each model family (e.g. lemmatization and stop-word removal for TF-IDF, BERTweet's official normalization rules).
 
### Models
- Logistic Regression and linear SVM (TF-IDF features, unigrams + bigrams)
- TextCNN (embedding layer + multi-size 1D convolutions)
- DistilBERT and BERTweet (fine-tuned via the HuggingFace `Trainer` API)
All five model families are trained across the three label strategies and two preprocessing regimes (30 experimental configurations), then evaluated for cross-dataset transfer to the Dell tweets dataset.
 
## Results
 
- Best SemEval dev result: **BERTweet, minimal preprocessing, A2 (Percentile Filtering) — 0.9005 macro-F1**.
- The percentile-based strategy (A2) is consistently competitive or best across model families, suggesting that removing ambiguous mid-range scores improves label quality.
- Transformers substantially outperform TF-IDF baselines and TextCNN.
- Cross-dataset transfer to the Dell dataset is strong for transformers, with BERTweet reaching up to 0.9672 macro-F1; linear models degrade more noticeably.
- Error analysis on the best model shows most mistakes involve sarcasm, political context, or ambiguous phrasing.
Full metrics for all 30 SemEval configurations and Dell cross-dataset results are available in the paper.
 
## Requirements
 
- Python 3
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `nltk`
- `scikit-learn`
- `torch`
- `transformers` (HuggingFace)
## Limitations
 
Reported numbers use the official SemEval dev split and may reflect some overfitting to development choices. Label engineering introduces noise from the underlying emotion annotations. The work is limited to English-language tweets, and transformer fine-tuning requires GPU resources.
 
## Ethical Considerations
 
Optimism/pessimism classifiers can be misused for profiling or manipulation if deployed without consent, and may reflect demographic or annotation biases present in Twitter data. This work is intended for aggregate analysis only, with uncertainty reporting and subgroup auditing recommended before any applied use.
