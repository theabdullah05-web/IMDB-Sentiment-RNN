# IMDB Sentiment Analysis using RNN

Binary sentiment classification of IMDB movie reviews using a **PyTorch RNN**, with **NLTK** text preprocessing and **TF-IDF** vectorization.

## Overview

This project trains a Recurrent Neural Network (RNN) to classify movie reviews as **positive** or **negative**. Raw reviews are cleaned and normalized with a classic NLP pipeline, converted into TF-IDF features, and then fed into an RNN built with PyTorch.

**Result:** ~85.9% accuracy on the held-out test set.

## Dataset

- **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50000-movie-reviews) (Kaggle)
- **Size:** 50,000 reviews (49,582 after removing duplicates)
- **Columns:** `review` (text), `sentiment` (`positive` / `negative`)

## Pipeline

1. **Data cleaning:** check for null values and drop duplicate reviews
2. **Text preprocessing:**
   - Convert to lowercase
   - Remove URLs
   - Remove punctuation
   - Remove HTML tags
   - Remove stopwords (NLTK)
   - Stemming (Porter Stemmer)
3. **Label encoding:** `positive` / `negative` mapped to `1` / `0`
4. **Vectorization:** TF-IDF with the top 5,000 features
5. **Train/test split:** 80% / 20% (`random_state=42`)
6. **Model:** single-layer RNN (hidden size 128) followed by a fully connected output layer
7. **Training:** Adam optimizer, binary cross-entropy loss, 10 epochs, batch size 64
8. **Evaluation:** accuracy on the test set

## Model Architecture

```
Input (5000 TF-IDF features)
        ↓
RNN (hidden_size=128, num_layers=1)
        ↓
Linear (128 → 1)
        ↓
Sigmoid → Positive / Negative
```

## Results

| Metric        | Value   |
|---------------|---------|
| Test accuracy | ~85.86% |

## Tech Stack

- Python
- PyTorch
- NLTK
- scikit-learn
- pandas

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/imdb-sentiment-rnn.git
cd imdb-sentiment-rnn
```

### 2. Install dependencies

```bash
pip install torch pandas scikit-learn nltk jupyter
```

### 3. Run the notebook

```bash
jupyter notebook RNN_for_SentimentAnalysis.ipynb
```

> **Note:** The notebook was written in Google Colab and mounts Google Drive to load the data. If running locally, remove the Drive-mounting cell and update the CSV path.

## Project Structure

```
imdb-sentiment-rnn/
├── IMDB Dataset.csv                  # Dataset
├── RNN_for_SentimentAnalysis.ipynb   # Full pipeline: preprocessing, training, evaluation
└── README.md
```

## License

This project is licensed under the MIT License.
