# Structural & Graph-Theoretic Analysis of AI vs Human Text (HC3)

This project investigates measurable structural differences between AI-generated and human-written text using the HC3 (Human–ChatGPT Comparison Corpus).

Rather than relying on black-box deep learning, the study focuses on interpretable linguistic features, graph-theoretic analysis, and linear models to evaluate structural separability.

---

## 🔍 Research Objective

To determine whether AI-generated text exhibits systematic structural and graph-based differences compared to human-written text.

We test the hypothesis that probabilistic next-token generation produces smoother and more centralized structural patterns.

---

## 📊 Dataset

- **Corpus:** HC3 (Human–ChatGPT Comparison Corpus)
- **Total responses:** 48,189
- **Paired sample (EDA):** 500 human / 500 AI
- **Evaluation:** QID-based hold-out split (no question leakage)

---

## 🧠 Feature Engineering

### Linguistic Features
- Word count
- Character count
- Unique word count
- Lexical diversity (Type-Token Ratio)
- Stopword ratio
- Pronoun ratio
- POS distribution (noun/verb/adjective ratios)

### Graph-Theoretic Features
Per-response token co-occurrence graphs:
- Degree centrality variance
- Clustering coefficient
- PageRank entropy
- Average shortest path length

---

## 📈 Modeling

All models evaluated on QID-based hold-out split.

| Model | ROC-AUC |
|--------|---------|
| Baseline Logistic Regression | 0.904 |
| Enriched Logistic Regression | 0.921 |
| Linear SVM | 0.920 |
| Character n-gram (3–5) | 0.998 |

Regularization stability verified via 5-fold GroupKFold (grouped by QID).

---

## 🔬 Interpretability

SHAP (SHapley Additive Explanations) applied to the enriched logistic regression model to identify key structural drivers.

Top discriminative features:
- Lexical diversity
- Word length
- POS ratios
- Stopword ratio

---

## 🏷 Keywords

NLP, AI Detection, Linguistic Analysis, Graph Theory, Interpretability, SHAP, Logistic Regression, SVM

---