import keras

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import LSTM, Embedding, Input
from keras.preprocessing.sequence import pad_sequences
from solution.dataman import minus_data
from keras.datasets import imdb

nb_classes = 5
nb_epoch = 5

tracex, tracey, tracet, target, label = minus_data()

max_trace = 0

for item in tracex:
    length = item.shape[0]
    if length > max_trace: max_trace=length

print(max_trace)
# max_features = 20000
# maxlen = 80  # cut texts after this number of words (among top max_features most common words)
# batch_size = 32
tracex = pad_sequences(tracex, maxlen=max_trace, dtype="float32")
tracey = pad_sequences(tracey, maxlen=max_trace, dtype="float32")


print('Build model...')
layer_x = Input(shape=(max_trace, ))
layer_y = Input(shape=(max_trace, ))

merged_vector = keras.layers.concatenate([layer_x, layer_y], axis=1)

# shared_lstm = LSTM(64)
#
# encoded_x = shared_lstm(layer_x)
# encoded_y = shared_lstm(layer_y)


hid_layer1 = Dense(256, activation="sigmoid")(merged_vector)
hid_layer2 = Dense(128, activation="sigmoid")(hid_layer1)
hid_layer3 = Dense(32, activation="sigmoid")(hid_layer2)

predictions = Dense(1, activation="sigmoid")(hid_layer3)

model = Model(inputs=[layer_x, layer_y], outputs=predictions)

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit([tracex, tracey],
          label,
          batch_size=256,
          epochs=100,
          validation_split=0.1)
score, acc = model.evaluate([tracex, tracey], label ,batch_size=32)
print('model:', model.summary())
print('Test score:', score)
print('Test accuracy:', acc)