import numpy as np
import pandas as pd
import wordninja
from scipy.sparse import hstack
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
stop_words = stopwords.words('english')
import re
from nltk.stem.porter import PorterStemmer
class pre_process:
    # https://stackoverflow.com/a/49146722/330558
    def _init_ (self, emoji_pattern):
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\u200d"
                           u"\u2640-\u2642" 
                           "]+", flags=re.UNICODE)
        
    
    def read_dataset(path):
        dataset = pd.read_csv(path, delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
        return dataset
    

    def remove_emoji(string):
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\u200d"
                           u"\u2640-\u2642" 
                           "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)
    
    

    def clean_text(dataset):

        corpus = []
        # Loop over the dataset for cleansing
        for i in range(0 , dataset.shape[0]):
            dataset ['text'][i]
            
            review = re.sub(r"http\S+", "", dataset ['text'][i])
            
            review = emoji_pattern.sub(r'', review)
            review = remove_emoji(review)
            review = " ".join([a for a in re.split('([A-Z][a-z]+)', review) if a])
            review = re.sub('[^a-zA-Z]' , ' ' , review)
            review = ' '.join(wordninja.split(review) )
            review = review.lower()
            review = re.sub(r"i'm", "i am",review)
            review = re.sub(r"he's", "he is",review)
            review = re.sub(r"she's", "she is",review)
            review = re.sub(r"that's", "that is",review)
            review = re.sub(r"where's", "where is",review)
            review = re.sub(r"what's", "what is",review)
            review = re.sub(r"\'ll", "will",review)
            review = re.sub(r"\'ve", "have",review)
            review = re.sub(r"\'re", "are",review)
            review = re.sub(r"won't", "will not",review)
            review = re.sub(r"can't", "can not",review)
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review) 
            corpus.append(review)
        
        # Drop empty rows & create dataframe with cleaned corpus
        df = pd.DataFrame({'text': corpus})
        #df['choose_one'] = dataset['choose_one']
        df['text'].replace('', np.nan, inplace=True)
        df.dropna(subset=['text'], inplace=True)
        #df['choose_one'].replace('', np.nan, inplace=True)
        #df.dropna(subset=['choose_one'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    
    
