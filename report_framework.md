# CDS525 Group Project Report Framework
## Fake News Detection using BiLSTM

> Target: ≥ 2000 words | Sections below include suggested content & word count

---

## 1. Introduction (~400 words)

### 1.1 Background
- Fake news is a growing problem on social media and online platforms.
- It can cause public panic, manipulate elections, and spread misinformation.
- Automated detection is essential due to the sheer volume of online content.

### 1.2 Problem Definition
- **Task**: Binary text classification — classify a news article as REAL (1) or FAKE (0).
- **Dataset**: fakenews.csv — 4,986 labeled news articles (text + label).
- **Metric**: Accuracy, Precision, Recall, F1-Score.

### 1.3 Current Popular Techniques
Briefly introduce the evolution of NLP for fake news detection:
- **Rule-based methods** (early): keyword matching, pattern detection — limited generalization.
- **Traditional ML**: Naive Bayes, SVM with TF-IDF features — fast but lacks context understanding.
- **Deep Learning**:
  - RNN / LSTM: captures sequential dependencies in text.
  - **BiLSTM**: bidirectional reading captures both past and future context. ✅ (Our choice)
  - Transformer/BERT: state-of-the-art but computationally heavy.
- Cite 2–3 relevant papers (see References section below for suggestions).

---

## 2. Design & Functions (~600 words)

### 2.1 Data Preprocessing Pipeline
Describe each step with brief justification:

| Step | Method | Reason |
|------|--------|--------|
| Lowercase | `str.lower()` | Normalize token case |
| Remove URLs | Regex | URLs add noise, not semantic value |
| Remove punctuation | Regex `[^a-z\s]` | Reduce vocabulary size |
| Tokenization | Whitespace split | Simple and effective for English |
| Vocabulary | Top 20,000 words | Covers most frequent tokens |
| Encoding | Index lookup + `<UNK>` | Convert text to model input |
| Padding | Fixed length 200 | Required for batch processing |
| Split | 70% train / 10% val / 20% test | Standard evaluation protocol |

### 2.2 Model Architecture: BiLSTM with Attention

```
Input Text
   ↓
[Embedding Layer]  dim=128, vocab=20,000
   ↓
[BiLSTM Layer]     hidden=256, layers=2, dropout=0.5
   ↓
[Attention Layer]  weighted sum over time steps
   ↓
[Dropout]          rate=0.5
   ↓
[Fully Connected]  256×2 → 1
   ↓
[Sigmoid]          output ∈ (0, 1) → threshold 0.5
```

**Why BiLSTM?**
- Standard LSTM only reads text left-to-right, missing right-to-left context.
- BiLSTM processes the sequence in both directions, capturing richer semantic features.
- **Attention mechanism** lets the model focus on the most informative words (e.g., sensational language in fake news).
- BiLSTM is computationally lighter than Transformer, suitable for a dataset of ~5,000 samples.

### 2.3 Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| BCEWithLogitsLoss | `−[y·log(σ(x)) + (1−y)·log(1−σ(x))]` | Baseline, numerically stable |
| Focal Loss | `−α(1−pt)^γ · log(pt)` | Handles class imbalance; down-weights easy examples |

**Parameters:**
- Focal Loss: α=0.25, γ=2.0 (standard values from Lin et al., 2017)

### 2.4 Optimizer & Hyperparameters

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| Optimizer | Adam | Adaptive learning rate, fast convergence |
| Learning Rate | 0.001 (baseline) | Standard starting point for Adam |
| Batch Size | 32 (baseline) | Balance between speed and gradient quality |
| Epochs | 15 | Sufficient for convergence on this dataset size |
| LR Scheduler | ReduceLROnPlateau | Reduce LR when validation loss plateaus |
| Gradient Clipping | 1.0 | Prevent exploding gradients in RNN |
| Dropout | 0.5 | Regularization to prevent overfitting |

---

## 3. Demonstration & Performance (~600 words)

### 3.1 Dataset Description
- **Total samples**: 4,986 news articles
- **Features**: Raw text (news headlines/articles)
- **Labels**: 0 = Fake, 1 = Real
- **Split**: ~3,490 train / ~390 val / ~980 test
- Describe any class imbalance observed and how it's handled.

### 3.2 Performance Visualization

Reference the 7 generated figures:

**Fig 1 – BCE Loss Baseline**
- Describe convergence trend.
- Comment on gap between train acc and test acc (overfitting?).

**Fig 2 – Focal Loss**
- Compare convergence speed vs BCE.
- If test acc improved, attribute to handling of hard examples.

**Fig 3 – Train Loss across Learning Rates**
- lr=0.1 typically diverges or oscillates.
- lr=0.001 provides smooth convergence.
- lr=0.0001 converges slowly but stably.

**Fig 4 – Test Accuracy across Learning Rates**
- Identify the best-performing learning rate.

**Fig 5 – Train Loss across Batch Sizes**
- Small batch (bs=8): noisy but may generalize better.
- Large batch (bs=128): smoother but may underfit.

**Fig 6 – Test Accuracy across Batch Sizes**
- Identify the optimal batch size.

**Fig 7 – Prediction Table (First 100 Test Samples)**
- Show the colored table (green = correct, red = wrong).
- Discuss any patterns in misclassified samples.

### 3.3 Final Performance Summary

| Model Config | Test Accuracy | Notes |
|-------------|--------------|-------|
| BiLSTM + BCE (lr=0.001, bs=32) | XX% | Baseline |
| BiLSTM + Focal (lr=0.001, bs=32) | XX% | Alt. loss |
| Best LR config | XX% | lr=? |
| Best BS config | XX% | bs=? |

*(Fill in actual numbers after running the code)*

---

## 4. Conclusion (~300 words)

### 4.1 Summary
- We implemented a **BiLSTM model with attention** for fake news detection.
- The model achieves ~XX% accuracy on the test set.
- Key findings:
  - Learning rate of 0.001 gives the best convergence.
  - Focal loss slightly outperforms BCE due to ...
  - Batch size of 32 provides the best trade-off.

### 4.2 Limitations
- The dataset is relatively small (~5,000 samples); larger datasets may improve generalization.
- BiLSTM cannot capture long-range global dependencies as well as Transformers.
- Our model only uses raw text; metadata (source, author, date) could improve accuracy.
- No pre-trained word embeddings (GloVe/Word2Vec) were used — this could boost performance.

### 4.3 Future Work
- Fine-tune BERT or RoBERTa for higher accuracy.
- Incorporate external knowledge graphs to detect factual errors.
- Extend to multi-class classification (e.g., satire, propaganda, misleading).
- Explore explainability methods (e.g., SHAP, LIME) to visualize attention weights.

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
2. Lin, T. Y., et al. (2017). Focal loss for dense object detection. *ICCV 2017*.
3. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL 2019*.
4. Shu, K., et al. (2017). Fake news detection on social media: A data mining perspective. *ACM SIGKDD*, 19(1), 22–36.
5. Zhou, X., & Zafarani, R. (2020). A survey of fake news: Fundamental theories, detection methods, and opportunities. *ACM Computing Surveys*, 53(5).

---

## Code Highlight (Key Snippets Only — DO NOT paste full code)

Include only these in the report:

1. **BiLSTM Model class** (forward method, ~15 lines)
2. **Attention mechanism** (3–5 lines)
3. **Focal Loss definition** (5–8 lines)
4. **Training loop summary** (pseudo-code or 10 lines max)
