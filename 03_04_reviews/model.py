from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import feature

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10_000)


def print_loss_graph(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


# Reviews
x_train = feature.vectorize_sequences(train_data)
x_test = feature.vectorize_sequences(test_data)
# Labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Set up model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10_000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile model with optimizer.
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model and get history
history = model.fit(
    x_train,
    y_train,
    epochs=4,
    batch_size=512,
    validation_data=(x_test, y_test))

results = model.evaluate(x_test, y_test)

model.save('./model.h5')

print(results)

