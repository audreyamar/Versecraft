from keras import Sequential
import tensorflow as tf
from keras.src.utils import pad_sequences
from keras.src.layers import Embedding, LSTM, Dropout, Dense
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.optimizers import Adam
import numpy as np
import pickle

import os


folder_path = 'Dataset'

poems = []
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            poems += file.readlines()
# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poems)

input_sequences = []
for line in poems:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,
                                         maxlen=max_sequence_len,
                                         padding='pre'))
# Create predictors and label
X, labels = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(labels, num_classes=len(tokenizer.word_index)+1)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(len(tokenizer.word_index)//2, activation='relu'))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01))
model.summary()

# Train the model
batch_size = 128
model.fit(X, y, epochs=30, batch_size=batch_size, verbose=1)

model.save('model.keras')

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
