import numpy as np
from keras.datasets import imdb

# Get review data ready for ML.


def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


def decode_review(sequence):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key)
                               for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence])
