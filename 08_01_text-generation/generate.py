import sys
import random
import numpy as np
from keras import models

import corpus

maxlen = 60
step = 3

model = models.load_model('./model.h5')

path = sys.argv[1]
# path = keras.utils.get_file(
#     'nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = corpus.text_from_path(path)
chars = corpus.get_chars(text)
char_indices = corpus.get_char_indices(chars)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, text, chars, char_indices):
    # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        generated_text = text[start_index: start_index + maxlen]

        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


generate_text(model, text, chars, char_indices)
