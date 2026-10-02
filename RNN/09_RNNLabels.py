sentences = [
    "food was good",
    "food was bad",
    "food was not good"
]

labels = [1,0,0]

for sentance , label in zip(sentences,labels):
    print("Sentance : ",sentance)
    print("Label : ",label)

    if label == 1:
        print("Meaning : Positive sentiment")
    else:
        print("Meaning : Negative sentiment")

    print("------------------------------")