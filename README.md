# CM3060 Natural Language Processing - University of London

Coursework for **CM3060 Natural Language Processing** (BSc Computer Science, University of London). The project compares traditional statistical text classifiers with an embedding-based deep learning model for **fake news detection**.

| Assessment | Project | Key topics |
|---|---|---|
| Mid Term | Fake news detection: statistical vs embedding models | Text preprocessing, TF-IDF, Word2Vec, Logistic Regression, SVM, LSTM |

**Tech:** Python · pandas · NLTK · scikit-learn · Gensim · TensorFlow / Keras · Jupyter

---

## Fake News Detection using Statistical and Embedding Models

**Question:** do complex embedding-based deep learning models actually beat simpler, more interpretable statistical models at detecting fake news?

### Dataset
- **Fake and Real News dataset** (Kaggle): `Fake.csv` (23,481 articles) and `True.csv` (21,417 articles)
- **44,898 articles** in total, roughly balanced between the two classes

### Pipeline
1. **Merge and label:** both files combined into one dataset with a binary label and saved locally
2. **Clean:** lowercasing, removing non-alphanumeric characters, and combining title and body text
3. **Tokenise and lemmatise:** NLTK tokenisation and the WordNet lemmatiser
4. **Two feature representations:**
   - **TF-IDF** with unigrams and bigrams for the statistical models
   - **Word2Vec** (100-dimension, trained on the corpus) averaged into document vectors, plus padded token sequences for the LSTM
5. **Models**
   - Logistic Regression (TF-IDF baseline)
   - Linear **SVM** (TF-IDF)
   - **LSTM** network (Embedding → LSTM(128) → sigmoid) with early stopping

### Results (held-out test set, 8,980 articles)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Linear SVM (TF-IDF)** | **0.9963** | **0.9958** | **0.9965** | **0.9961** |
| Logistic Regression (TF-IDF) | 0.9866 | 0.9806 | 0.9916 | 0.9861 |
| LSTM (embeddings) | 0.9771 | 0.9855 | 0.9662 | 0.9757 |

### Key findings
- **The simple model won.** Linear SVM on TF-IDF beat the LSTM on every metric, with only about 30 errors out of 8,980 articles.
- **The LSTM costs more.** Each epoch took about 80 seconds, compared with seconds for the SVM, and its decisions are much harder to interpret.
- **Why:** on this dataset, word-frequency signals are already highly discriminative. The result matches findings in the literature that well-tuned traditional models stay competitive for text classification.

---

## How to run

```bash
pip install pandas numpy nltk scikit-learn gensim tensorflow jupyter
python -m nltk.downloader punkt wordnet stopwords
jupyter notebook
```
Download `Fake.csv` and `True.csv` from Kaggle ("Fake and Real News Dataset") into a `dataset/` folder next to the notebook. They are not included here because of file size.
