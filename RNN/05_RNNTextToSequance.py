from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "food was good",
    "food was bad",
    "food was not good"
]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

sequances = tokenizer.texts_to_sequences(sentences)

for sentance, sequance in zip(sentences,sequances):
    print("Sentance : ",sentance)
    print("Sequance : ",sequance)
    print("--------------------------------")