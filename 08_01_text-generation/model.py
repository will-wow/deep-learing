import sys
import random
import keras
import numpy as np
from keras import layers
import generate

import corpus

maxlen = 60
step = 3

path = sys.argv[1]
path = keras.utils.get_file(
    'nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = corpus.text_from_path(path)

sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('number of sequences', len(sentences))

chars = sorted(list(set(text)))
char_indices = dict((char, chars.index(char)) for char in chars)

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


for epoch in range(1, 60):
    print('epoch', epoch)

    model.fit(x, y, batch_size=128, epochs=1)
    generate.generate_text(model, text, chars, char_indices)

model.save('./model.h5')

print('done training!')
