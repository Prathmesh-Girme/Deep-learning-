"""
Movie Review Sentiment Analysis using LSTM (IMDB dataset)

Pipeline:
    1. Load the IMDB dataset
    2. Decode and display sample reviews
    3. Pad the reviews to equal length
    4. Build the model: Embedding -> LSTM -> Dense (sigmoid)
    5. Train and evaluate the model
    6. Predict the sentiment of one test review
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
VOCAB_SIZE = 10000          # Keep the 10,000 most frequent words
MAX_LENGTH = 200            # Use only the first 200 words of each review
EMBEDDING_DIM = 32          # Each word is represented by 32 numbers
LSTM_UNITS = 64             # Size of the LSTM hidden state
EPOCHS = 3
BATCH_SIZE = 64
TEST_REVIEW_NUMBER = 0      # Which test review to predict

LINE = "_" * 50

# IMDB reserves indexes 0, 1, 2 for padding, start and unknown tokens.
# Real words therefore start from index 3.
INDEX_OFFSET = 3


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def build_reverse_word_index():
    """Create a dictionary that maps number -> word."""
    word_index = imdb.get_word_index()
    return {index + INDEX_OFFSET: word for word, index in word_index.items()}


def decode_review(encoded_review, reverse_word_index):
    """Convert a list of numbers back into readable text."""
    words = [
        reverse_word_index.get(number, "?")
        for number in encoded_review
        if number >= INDEX_OFFSET
    ]
    return " ".join(words)


def sentiment_name(value):
    """Convert 1/0 (or a probability) into a label."""
    return "POSITIVE" if value >= 0.5 else "NEGATIVE"


def build_model():
    """Embedding -> LSTM -> Dense (sigmoid)."""
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        LSTM(units=LSTM_UNITS),
        Dense(units=1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
def main():
    print(LINE)
    print("Movie Review Sentiment Analysis using LSTM")
    print(LINE)

    # Step 1 : Load the dataset (0 = negative, 1 = positive)
    print("Loading the dataset...")
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    print("Number of training reviews :", len(X_train))
    print("Number of testing reviews  :", len(X_test))

    # Step 2 : Show sample reviews
    reverse_word_index = build_reverse_word_index()

    print(LINE)
    print("Sample reviews")
    print(LINE)

    for i in range(4):
        print("Review number :", i + 1)
        print(decode_review(X_train[i], reverse_word_index))
        print("Sentiment :", sentiment_name(Y_train[i]))
        print(LINE)

    # Step 3 : Padding
    # "pre" padding puts the zeros at the start, so the LSTM finishes on
    # real words instead of zeros. This works better than "post" padding.
    X_train_padded = pad_sequences(
        X_train, maxlen=MAX_LENGTH, padding="pre", truncating="post"
    )
    X_test_padded = pad_sequences(
        X_test, maxlen=MAX_LENGTH, padding="pre", truncating="post"
    )

    print("Training data shape :", X_train_padded.shape)
    print("Testing data shape  :", X_test_padded.shape)

    # Step 4 : Build the model
    model = build_model()
    model.build(input_shape=(None, MAX_LENGTH))
    model.summary()

    # Step 5 : Train the model
    print("Model training...")
    model.fit(
        X_train_padded,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
    )
    print("Model training completed")

    # Step 6 : Evaluate the model
    loss, accuracy = model.evaluate(X_test_padded, Y_test, verbose=0)

    print("Testing loss     :", loss)
    print("Testing accuracy :", accuracy)

    # Step 7 : Predict one review
    review_text = decode_review(X_test[TEST_REVIEW_NUMBER], reverse_word_index)

    print(LINE)
    print("Review given to the model")
    print(LINE)
    print(review_text)

    actual_sentiment = sentiment_name(Y_test[TEST_REVIEW_NUMBER])

    sample = X_test_padded[TEST_REVIEW_NUMBER : TEST_REVIEW_NUMBER + 1]
    probability = float(model.predict(sample, verbose=0)[0][0])
    predicted_sentiment = sentiment_name(probability)

    # Step 8 : Final result
    print(LINE)
    print("Final result")
    print(LINE)
    print("Prediction probability :", round(probability, 4))
    print("Actual sentiment       :", actual_sentiment)
    print("Predicted sentiment    :", predicted_sentiment)


if __name__ == "__main__":
    main()
