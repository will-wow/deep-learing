import numpy as np
from keras import models
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence

import feature

word_index = imdb.get_word_index()

skip_top = 0
num_words = 10_000
start_char = 1
oov_char = 2
index_from = 3


def predict(reviews):
    model = __load_model()
    word_lists = map(text_to_word_sequence, reviews)
    indicies = map(review_to_indicies, word_lists)
    sequences = no_out_of_value_indicies(indicies)
    vectors = feature.vectorize_sequences(sequences)
    return model.predict(np.array(vectors))


def review_to_indicies(words):
    indicies = [
        word_index[word] if word in word_index else 0 for word in words
    ]

    return np.array(indicies)


def no_out_of_value_indicies(xs):
    xs = [[start_char] + [w + index_from for w in x] for x in xs]

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
          for x in xs]

    return xs


def word_to_index(word, num_words):
    index = word_index[word] if word in word_index else 0
    return index + 3 if index < num_words else 2


def __load_model():
    return models.load_model('./model.h5')


good = "The best way I can describe John Wick is to picture Taken but instead of Liam Neeson it's Keanu Reeves and instead of his daughter it's his dog. That's essentially the plot of the movie. John Wick Reeves is out to seek revenge on the people who took something he loved from him. It's a beautifully simple premise for an action movie - when action movies get convoluted, they get bad A Good Day to Die Hard. John Wick gives the viewers what they want: Awesome action, stylish stunts, kinetic chaos, and a relatable hero to tie it all together. John Wick succeeds in its simplicity"
bad = "I know it is too late to warn you not to go see this movie. But still, I am writing this review to express my disappointment, and to say I have no idea why this movie can be rate 7+ here. There must be something wrong. I am a big fan of Keanu. And I do not mind action movie with simple plot, as long as it looks cool. Nevertheless, I was brought down big time. The plot is unbelievably ridiculous. It will leave you amazed, wondering if whether someone pro-read the script before they made it a movie. No twist, no surprise, no character development, nothing thrilling or exciting whatsoever. It is utter boredom. I felt intelligently insulted, to be honest."

print(predict([good, bad]))
