from disaster_pipeline import Disaster
from text_pre_proccessing import pre_process
import numpy as np
import pickle
import pandas as pd
from collections import Counter


from tqdm import tqdm
stop_words = stopwords.words('english')
from nltk import word_tokenize
from sklearn.utils import shuffle



# =============================================================================
# Load disasters dataset
# =============================================================================
dataset_original =  Disaster.load_dataset('Dataset/Clean_Disasters_T_79187_.csv')
#dataset_original["index"] = 0
#dataset_original.drop(['Disaster', 'Geo', 'is_duplicate','Unnamed: 0','language'], axis=1,inplace = True)
dataset =  pre_process.clean_text(dataset_original)
#dataset = pd.read_csv('Dataset/earth/dataset_clean_nepal.csv',delimiter = ',',nrows=1000 ,converters={'text': str}, encoding = "ISO-8859-1")
dataset = shuffle(dataset)
# =============================================================================
# Dataset Pre-process for the machine learning models
# 
# =============================================================================

# =============================================================================
# # Load glove embedings matrix
# 
# # Convert tokens to embeddings values
# =============================================================================

text = dataset.text
#label = dataset['choose_one']
data_concatenated = pd.concat([text],axis=1)

data_concatenated.head()



embeddings_index = {}
f = open('embeddings/glove.twitter.27B.200d.txt', encoding="utf8")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    try:
        #coefs = np.asarray(values[1:], dtype='float32')
        coefs = np.asarray(values[1:], dtype='float64')
        embeddings_index[word] = coefs
    except ValueError:
        pass
f.close()
print('Found %s word vectors.' % len(embeddings_index))




def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return (v / np.sqrt((v ** 2).sum())).astype('Float64')

print('Checkpoint2 -Normalized Vector for Sentences are created')

data_concatenated.text = data_concatenated.text.apply(sent2vec)


#df = pd.DataFrame(index=np.arange(0, data_concatenated.shape[0]), columns=[np.arange(0,50)])

data_concatenated.head()

res = []
#lbl = []
for index, row in data_concatenated.iterrows(): 
    #df[index] = [row.reshape(-1, len(row)) for n in range(50)]
    #df.append(pd.DataFrame(row.reshape(-1, len(row))))
    res.append(pd.DataFrame(row['text'].reshape(-1, len(row['text']))))
    #lbl.append(row['choose_one'])
data_concatenated = pd.concat(res, axis=0)

#data_concatenated['label'] = lbl

data_concatenated.dropna(axis='columns', inplace = True)

data_concatenated.reset_index(inplace = True)

data_concatenated.head(22)

# Split features and Target Variable
#y_df = data_concatenated['label']
#X_df = data_concatenated.drop('label', axis = 1)

features = data_concatenated.values

# =============================================================================
# Load the machine learning models from disk
# =============================================================================
models_1 = Disaster.load_ml_models_1()
models_2 = Disaster.load_ml_models_2()




# =============================================================================
# Result evaluation
# =============================================================================
earthquake = []
time_earth = []
hurricane = []
time_hurri = []
irrelevant = []
time_irrelevant = []
for index in range(0,len(features)):  
    predictions_labels = []
    predictions_proba = []
    for model in models_1:
         # Predict disaster
         predictions_labels.append(float(model.predict(features[index].reshape(1, -1))))
         predictions_proba.append(np.max(model.predict_proba(features[index].reshape(1, -1))))
        
    # Voting & Classify Disaster
    if Counter(predictions_labels)[1.0] > Counter(predictions_labels)[0.0]:
        Disaster.disaster_recognition(models_2,earthquake,hurricane,time_earth,time_hurri,features[index],dataset_original.text[index],dataset_original.Date[index])

    elif Counter(predictions_labels)[1.0] == Counter(predictions_labels)[0.0]:
        print("draw disaster prediction")
        lbl_1 = 0
        lbl_0= 0
        for idx,lbl in enumerate(predictions_labels):
            if lbl == 1.0 :
                lbl_1 += predictions_proba[idx]
            else:
                lbl_0 += predictions_proba[idx]
        
        
        if (lbl_1/Counter(predictions_labels)[1.0]) > (lbl_0/Counter(predictions_labels)[0.0]):
            Disaster.disaster_recognition(models_2,earthquake,hurricane,time_earth,time_hurri,features[index],dataset_original.text[index],dataset_original.Date[index])
        
        
    else:
        print('Irrelevant tweet')
        irrelevant.append( dataset_original.text[index] ) 
        time_irrelevant.append( dataset_original.Date[index] )
        #irrelevant.append(dataset_original.text[index])
        
        
earth_d = {'Date':time_earth,'text':earthquake}   
earth_df = pd.DataFrame(earth_d)
earth_df.to_csv("300k_earth_df.csv",sep=',',encoding='utf-8')
##################################################################
hurri_d = {'Date':time_hurri,'text':hurricane}   
hurri_df = pd.DataFrame(hurri_d)
hurri_df.to_csv("300k_hurri_df.csv",sep=',',encoding='utf-8')
################################################################
irrelevant_d ={'Date':time_irrelevant,'text':irrelevant}
irrelevant_df = pd.DataFrame(irrelevant_d)
irrelevant_df.to_csv("300k_irre_df.csv",sep=',',encoding='utf-8')
################################################################################


# =============================================================================
# Evaluate time
# =============================================================================
#def plot_time_hist():
#    ufo =pd.read_csv('Dataset/earth/Result_1M_earth_df.csv',delimiter = ',' ,converters={'text': str}, encoding = "utf-8",engine='python' )
#    ufo.head()
#    ufo.dtypes
#    ufo['Date'] = pd.to_datetime(ufo.Date)
#    ufo.dtypes
#    ufo.Date.value_counts().sort_index().plot()
#
#
#plot_time_hist()











#





