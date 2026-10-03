# LSTM - Movie Review Sentiment Analysis

Classifies IMDB movie reviews as **positive** or **negative** using an LSTM network built with Keras.

## Pipeline

1. Load the IMDB dataset (25,000 training and 25,000 test reviews)
2. Decode numbers back into readable text and show sample reviews
3. Pad every review to 200 words
4. Build the model
5. Train and evaluate
6. Predict the sentiment of one test review

## Model

```
Review -> Embedding(10000, 32) -> LSTM(64) -> Dense(1, sigmoid) -> Positive / Negative
```

| Setting | Value |
|---|---|
| Vocabulary size | 10,000 words |
| Review length | 200 words |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Epochs / batch size | 3 / 64 |

## Why LSTM instead of a simple RNN

A simple RNN forgets early words in long sentences (the vanishing gradient problem). An LSTM keeps a separate cell state with gates that decide what to remember and what to forget, so it handles long reviews much better.

## Run

```bash
pip install tensorflow
python LSTM_IMDB_Sentiment.py
```

The dataset downloads automatically the first time you run it.
