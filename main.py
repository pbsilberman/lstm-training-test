# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# load ascii text
filename = "trump.txt"
raw_text = open(filename, encoding = "ISO-8859-1").read()


import spacy
nlp = spacy.load('en')
text = raw_text[0:999837]
doc = nlp(text)


# split the raw text up by word using spacy
sent_struct = []
sent = []
pos= []
words = []
word_dict = {}

for token in doc:
    if token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and 'http' not in token.text:
        sent.append(token.pos_)
        pos.append(token.pos_)
        words.append(token.text)
        key = token.text
        if key not in word_dict:
            word_dict[token.text] = token.pos_
    if token.pos_ == 'SPACE':
        sent_struct.append(sent)
        sent = []


# create mapping of unique wards to integers, and a reverse mapping
words_clean = sorted(set(words))
words_to_int = dict((w, i) for i, w in enumerate(words_clean))
int_to_words = dict((i, w) for i, w in enumerate(words_clean))


# summarize the loaded data
n_words = len(words)
n_vocab = len(words_clean)
print("Total Words: ", n_words)
print("Total Vocab: ", n_vocab)


# prepare the dataset of input to output pairs encoded as integers
seq_length = 15
dataX = []
dataY = []
for i in range(0, n_words - seq_length, 1):
    seq_in = words[i:i + seq_length]
    seq_out = words[i + seq_length]
    dataX.append([words_to_int[word] for word in seq_in])
    dataY.append(words_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# define the checkpoint
filepath="trump-check-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)

