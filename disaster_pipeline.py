

import pickle
from tqdm import tqdm
# Pandas, sklern, numpy, matplotlib, other
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.externals import joblib
from collections import Counter
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Flatten, Input, LSTM, Embedding, Dropout, Activation,Bidirectional, GlobalMaxPool1D
from keras.models import Model,Sequential,load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import model_from_json
from keras.callbacks import History,EarlyStopping,ModelCheckpoint
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

#Word processing libraries
import re
import nltk 
import sklearn 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk import punkt
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')
from sklearn.utils import shuffle

# NLTK
import nltk
from nltk.corpus import stopwords

from text_pre_proccessing import pre_process

class Disaster:
  
    def _init_(self, max_len):
        self.max_len = max_len
       
    @staticmethod
    def process(path):
        dataset = pre_process.read_dataset(path)
        cleaned_dataset = pre_process.clean_text(dataset)
        
        return cleaned_dataset
    
    @staticmethod
    def load_dataset(path):
        dataset = pd.read_csv(path,delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
        dataset = shuffle(dataset)
        return dataset #list_sentences, list_labels


    
    @staticmethod
    def load_ml_models_1():
        models = []
        models.append(joblib.load('model_1/LRclassifier.joblib.pkl'))
        models.append(joblib.load('model_1/AdaBoostclassifier.joblib.pkl'))
        models.append(joblib.load('model_1/BgCclassifier.joblib.pkl'))
        models.append(joblib.load('model_1/NBclassifier.joblib.pkl'))
        
        return models
    
    @staticmethod
    def load_ml_models_2():
        models = []
        models.append(joblib.load('model_2/BgCclassifier.joblib.pkl'))
        models.append(joblib.load('model_2/NBclassifier.joblib.pkl'))
        models.append(joblib.load('model_2/LRclassifier.joblib.pkl'))
        return models

    
    @staticmethod
    def load_model_1(path):
        # load json and create model
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_1 = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model_1.load_weights("model_1/model_1.best.hdf5")
        print("Loaded model 1 from disk")
        
        return loaded_model_1
    def disaster_recognition(models_2,earthquake,hurricane,time_earth,time_hurri,datapoint,dataset_original_point,date_point):
        predictions_2 = [] 
        
        # Classify Disaster
        for model_2 in models_2:
            predictions_2.append(float(model_2.predict(datapoint.reshape(1, -1))))
            
        if Counter(predictions_2)[1.0] > Counter(predictions_2)[0.0]:
            print('erthquake')
            earthquake.append( dataset_original_point ) 
            time_earth.append( date_point )
        else:
            print('Hurricane')
            hurricane.append( dataset_original_point ) 
            time_hurri.append( date_point )
        return earthquake,hurricane,time_earth,time_hurri