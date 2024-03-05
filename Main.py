import pickle
import tensorflow as tf
import numpy as np
from keras.src.utils import pad_sequences
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

# Load the pre-trained model
model = tf.keras.models.load_model('model.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# You need to know the max_sequence_len used during training
# If it's not saved, you can set it manually (as used in training)
max_sequence_len = 100


def generate_poem(seed_text, next_words=20, temperature=1.0, words_per_line=5):
    poem = seed_text
    words = seed_text.split()
    nbr = 0

    while nbr <= next_words:
        token_list = tokenizer.texts_to_sequences([poem])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predictions = model.predict(token_list, verbose=0).astype('float64')

        # Apply temperature sampling
        predictions = np.log(predictions + 1e-10) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probabilities = np.random.multinomial(1, predictions[0], 1)
        predicted_word_index = np.argmax(probabilities)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break

        # Avoid repetition and add new word
        if output_word not in words[-3:]:
            words.append(output_word)
        else:
            continue

        nbr = len(words)
        # Add a new line every 'words_per_line' words
        poem += " "
        poem += output_word
        if len(words) % words_per_line == 0:
            poem += "\n"
        nbr = nbr + 1
    return poem


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def home():
    seed_text = request.args.get('poem')
    nbr_line = int(request.args.get('line'))
    words_per_line = int(request.args.get('words'))
    craziness = float(request.args.get('crazy'))
    poem = generate_poem(seed_text, nbr_line * words_per_line, craziness, words_per_line)
    return poem


if __name__ == '__main__':
    app.run(debug=True)
