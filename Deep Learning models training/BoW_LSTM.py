
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
DATA_FILE=f'{path}Clean_Disasters_T_79187_.csv'
TRAIN_DATA_FILE=f'{path}train.csv'
TEST_DATA_FILE=f'{path}Clean_MODEL2__Earth_Hurr_for_27434.csv'


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk 
import sklearn 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#import pattern3
#from pattern3.en import lemma
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer

print('The scikit-learn version is {}.'.format(sklearn.__version__))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_dataset():
    dataset = pd.read_csv('data/Clean_Disasters_T_79187_.csv',delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
    return dataset

def class_hist(dataset):
    ##Class histgram
    dataset['choose_one'].hist()
    # Class percentage
    perc = dataset.choose_one.value_counts(normalize=True)
    # Provide descriptive statistics 
    pd.plotting.scatter_matrix(pd.DataFrame(dataset), alpha = 0.9, figsize = (8,6), diagonal = 'kde')
    return perc
    
def missing_values(dataset):    
    # Check for missing values
    print('Missing Values')
    display(dataset.isnull().sum())

def make_corpus():    
    corpus = []
    for i in range(0,79187):
        corpus.append(dataset.text[i])
    return corpus
 
def Bow_Split(corpus,dataset,max_features): #### 2-Bag of words model
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    #Count Vecorizer
    cv = CountVectorizer(max_features = (max_features))
    X = cv.fit_transform(corpus).toarray() 
    
    ####Tf-Idf Vectorizer
    #tf = TfidfVectorizer(max_features=(50))
    #X = tf.fit_transform(corpus).toarray()
    
    #Split Dataset to X and y
    y = dataset.iloc[: , 3].values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    return X,y


def Test_Train_Split(X,y,test_size = 0.3): #Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42,shuffle=True,stratify=dataset['choose_one'].values)
    return X_train, X_test, y_train, y_test


dataset = read_dataset()
class_hist(dataset)
missing_values(dataset)
corpus = make_corpus()
X,y= Bow_Split(corpus,dataset,max_features=50)
X_train, X_test, y_train, y_test = Test_Train_Split(X,y,test_size = 0.3)


# In[4]:


dataset.index


# In[ ]:


"""
seq_length = X.shape[1]
# vocabulary size
vocab_size = 79187 + 1
# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
"""


# In[ ]:


"""
# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
#model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax'))
print(model.summary())
"""


# In[1]:


seq_length = X.shape[1]
# vocabulary size
vocab_size = 79187 + 1
maxlen = 50 # max number of words in a comment to use
inp = Input(shape=(maxlen,))
x = Embedding(vocab_size, 50,kernel_initializer='random_uniform')(inp)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
#x = BatchNormalization()(x)
x = GlobalMaxPool1D()(x)

#x = Dense(64, activation="relu")(x)
#x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[18]:


history = History()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
save_best = ModelCheckpoint('model_3.hdf', save_best_only=True, 
                           monitor='val_loss', mode='min')
history = model.fit(X_train, y_train, batch_size=150, validation_data=(X_test, y_test),
                    epochs=10, verbose=1,callbacks=[early_stopping,save_best,history])


# In[19]:


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

