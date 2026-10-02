import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocesssing.sequence import pad_sequences
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

#step 1 : Load the data
train_sentences = [
    "Food was good",
    "Food was bad",
    "Food was excellent", 
    "Food was terrible",
    "Service was good",
    "Service was bad",
    "Service was excellent",
    "Service was terrible",
    "ambience was good",
    "ambience was bad",
    "ambience was excellent",
    "ambience was terrible"
]



train_labels = [
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0
    
    
]

# step 2 : Tokenization

tokenizer = Tokenizer(OOV_token = "<OOV>")

tokenizer.fit_on_texts(train_sentences)

#step 3 : Convert training data into sequence


train_sequence = tokenizer.texts_to_sequences(train_sentences)

print("Training sequences : ")

for sentence , sequence in zip(train_sentences,train_sequence):
    
    print(sentence, " ->", sequence)
    
    #step 4: Apply padding
    
    max_length  = 4
    
    X_train = pad_sequences(
        train_sequence,
        maxlen = max_length,
        padding = "pre"
        
        
        
)
    
    Y_Train = np.array(train_labels)
    
    print("padded training data")
    print(X_train)
    
    print("Training labels :")
    print(Y_Train)
    
    
