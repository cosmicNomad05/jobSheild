# JobShield — Fake Job Posting Detection System

An NLP + ML web application that detects fraudulent job postings using
Logistic Regression trained on TF-IDF features.

---

## Project Structure

```
fakejob/
├── app.py                  ← Flask web application (main entry point)
├── train.py                ← Model training script (run once)
├── requirements.txt        ← Python dependencies
├── model/                  ← Generated model files (after training)
│   ├── model.pkl
│   └── vectorizer.pkl
├── data/                   ← Place your CSV dataset here
│   └── fake_job_postings.csv
└── templates/
    ├── base.html           ← Shared layout + nav
    ├── index.html          ← Homepage + input form
    ├── result.html         ← Analysis results page
    └── about.html          ← About page

```

---

## How It Works

```
User Input
    │
    ▼
Text Cleaning          lowercase, strip HTML, normalize whitespace
    │
    ▼
TF-IDF Vectorization   5,000 features, unigrams+bigrams, sublinear_tf=True
    │
    ▼
Logistic Regression    predict_proba() → fraud probability 0.0–1.0
    │
    ├── Signal Extraction   top fraud/genuine words by coefficient × tfidf
    └── Domain Rule Engine  email domain whitelist/suspicious pattern check
    │
    ▼
Results Page           Score gauge + risk badge + signals + domain verdict
```

---

## Risk Tiers

| Score       | Tier          | Meaning                                |
|-------------|---------------|----------------------------------------|
| 0–35%       | 🟢 LOW        | No strong fraud signals                |
| 35–70%      | 🟡 MEDIUM     | Some suspicious signals — caution      |
| 70–100%     | 🔴 HIGH       | Strong fraud indicators — avoid        |

---

## Model Details

| Parameter           | Value                        |
|---------------------|------------------------------|
| Dataset             | EMSCAD (17,880 job listings) |
| Algorithm           | Logistic Regression          |
| Class balancing     | class_weight='balanced'      |
| Vectorizer          | TF-IDF                       |
| Max features        | 5,000                        |
| N-gram range        | (1, 2) — unigrams + bigrams  |
| Train/test split    | 80/20, stratified            |
| Evaluation metrics  | Precision, Recall, F1, AUC   |

---

## Ethical Notes

- This is an academic prototype. Do not use as the sole basis for decisions.
- The model may miss novel scam patterns not present in the training data.
- No user data is stored. All analysis is performed in memory.