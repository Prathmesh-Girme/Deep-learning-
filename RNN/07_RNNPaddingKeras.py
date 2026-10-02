from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "food was good",
    "food was bad",
    "food was not good"
    ]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

sequances = tokenizer.texts_to_sequences(sentences)

print("Original Sequances. ")
for sequance in sequances:
    print(sequance, "Length : ", len(sequance))

print("All sequances are of diffrent lengths")

max_length = 4

padded_sequances = pad_sequences(
    sequances,
    maxlen = max_length,
    padding = "pre"
)

for sentance, sequqnce, padded in zip(sentences,sequances,padded_sequances):
    print("Sentance : ",sentance)
    print("Original sequance : ",sequqnce)
    print("Padded sequance : ",padded)
    print("----------------------------------------")