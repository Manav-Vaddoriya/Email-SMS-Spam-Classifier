
## Project Overview

This project classifies incoming Email/SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) and Machine Learning techniques. It is designed with **reproducibility, scalability** in mind.

Key highlights:

* NLP-based text preprocessing
* TF-IDF feature extraction
* Machine Learning model training
* Experiment tracking with DVC
* CI-enabled testing
* Streamlit-based web application

---

## Project Structure

```
.
├── .dvc/                    # DVC metadata
├── .github/workflows/       # CI pipelines (GitHub Actions)
├── dvclive/                 # Experiment logs
├── experiments/             # Experiment scripts
├── src/                     # Core ML & preprocessing logic
├── tests/                   # Unit tests
├── app.py                   # Streamlit application
├── dvc.yaml                 # DVC pipeline definition
├── dvc.lock                 # Pipeline lock file
├── params.yaml              # Model & pipeline parameters
├── model.pkl                # Trained ML model
├── vectorizer.pkl           # TF-IDF vectorizer
├── .gitignore
├── .dvcignore
└── README.md
```

---

## ML & NLP Pipeline

1. **Text Preprocessing**

   * Lowercasing
   * Tokenization
   * Removal of stopwords and punctuation
   * Stemming using NLTK

2. **Feature Engineering**

   * TF-IDF Vectorization

3. **Model Training**

   * Supervised ML classifier trained on labeled spam/ham data

4. **Evaluation**

   * Accuracy, Precision, Recall, F1-score

---

## Experiment Tracking (DVC)

* Dataset, model, and experiments are tracked using **DVC**
* Pipelines defined in `dvc.yaml`
* Parameters managed via `params.yaml`
* Ensures **full reproducibility** across environments

Run pipeline:

```bash
dvc repro
```

---

## Continuous Integration

* Integrated **GitHub Actions** for CI
* Automatically runs unit tests on every push
* Prevents broken code from being merged

---

## Web Application (Streamlit)

The project includes an interactive **Streamlit app** that allows users to:

* Enter an Email/SMS message
* Instantly classify it as **Spam** or **Not Spam**

Run the app locally:

```bash
streamlit run app.py
```

---

## Tech Stack

* **Language:** Python
* **Libraries:** Scikit-learn, NLTK, Streamlit
* **MLOps Tools:** DVC, Git, GitHub Actions
* **Concepts:** NLP, ML Pipelines, CI/CD, Model Versioning

---

## Key Learnings

* Built a reproducible ML pipeline using DVC
* Gained hands-on experience with NLP preprocessing
* Implemented CI for ML workflows

---

## Future Improvements

* Add model comparison (Logistic Regression, SVM)
* Include ROC-AUC and confusion matrix visualization
* Containerize using Docker

---
## Demo Video
* https://youtu.be/gwenJZSz_Ds
