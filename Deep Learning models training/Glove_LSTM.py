
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys, os, re, csv, codecs, numpy as np, pandas as pd
 
    
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import to_categorical
from keras.regularizers import l2,l1
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.callbacks import History,EarlyStopping,ModelCheckpoint

# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.manifold import TSNE


# In[2]:


path = 'data/'

EMBEDDING_FILE=f'glove.6B.300d.txt'
DATA_FILE=f'{path}Clean_MODEL2__Earth_Hurr_for_27434.csv'
TRAIN_DATA_FILE=f'{path}train.csv'
TEST_DATA_FILE=f'{path}Clean_MODEL2__Earth_Hurr_for_27434.csv'


# In[3]:


embed_size = 300 # how big is each word vector
max_features = 300 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a comment to use


# In[4]:


dataset = pd.read_csv(DATA_FILE,delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
dataset['choose_one'].replace([2,1],[1,0],inplace=True)
X = dataset.text
y = dataset.choose_one
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train = pd.concat([X_train, y_train], axis = 1)


# In[5]:


list_sentences_train = X_train.values#train["text"].fillna("_na_").values
#list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = y_train.values#train['choose_one'].values
list_sentences_test = X_test.values#test["text"].fillna("_na_").values
y_test = y_test.values#test['choose_one'].values


# In[6]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[7]:


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE ,encoding='utf8'))


# In[8]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[9]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[10]:


"""
# cnn model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

model = Sequential()
e = Embedding(max_features, embed_size, weights=[embedding_matrix], 
              input_length=300, trainable=False)

model.add(Conv1D(64, 3, activation='relu'))
#model.add(MaxPooling1D(3))
model.add(Dropout(0.2))

model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


# compile the model
Adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=Adam_opt, loss='binary_crossentropy', metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
save_best = ModelCheckpoint('toxic.hdf', save_best_only=True, 
                           monitor='val_loss', mode='min')

history = History()
history = model.fit(X_t, y_train, batch_size=100, validation_data=(X_te, y_test),
                    epochs=100, verbose=1,callbacks=[early_stopping,save_best])
"""   


# In[11]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
#x = BatchNormalization()(x)
x = GlobalMaxPool1D()(x)

#x = Dense(64, activation="relu")(x)
#x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[12]:


history = History()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
save_best = ModelCheckpoint('model_2.hdf', save_best_only=True, 
                           monitor='val_loss', mode='min')
history = model.fit(X_t, y_train, batch_size=150, validation_data=(X_te, y_test),
                    epochs=10, verbose=1,callbacks=[early_stopping,save_best,history])



#model.fit(X_t, y_train, batch_size=32, validation_data=(X_te, y_test), epochs=100, validation_split=0.2);


# In[13]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

