import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
from sklearn.model_selection import train_test_split
stop_words = stopwords.words('english')
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.stem.porter import PorterStemmer
from keras.callbacks import History,EarlyStopping,ModelCheckpoint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def read_dataset():
    df = pd.read_csv('Clean_Disasters_T_79187_.csv',delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
    return df
df = read_dataset()
text = df.text
label = df['choose_one']
data_concatenated = pd.concat([text, label],axis=1)

data_concatenated.head()



embeddings_index = {}
f = open('glove.twitter.27B.200d.txt', encoding="utf8")
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
lbl = []
for index, row in data_concatenated.iterrows(): 
    #df[index] = [row.reshape(-1, len(row)) for n in range(50)]
    #df.append(pd.DataFrame(row.reshape(-1, len(row))))
    res.append(pd.DataFrame(row['text'].reshape(-1, len(row['text']))))
    lbl.append(row['choose_one'])
data_concatenated = pd.concat(res, axis=0)

data_concatenated['label'] = lbl

data_concatenated.dropna(axis='columns', inplace = True)

data_concatenated.reset_index(inplace = True)

data_concatenated.head(22)

# Split features and Target Variable
y_df = data_concatenated['label']
X_df = data_concatenated.drop('label', axis = 1)


# Feature - Target split
features = X_df.values
target = y_df.values
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.ensemble import RandomForestClassifier

#xtrain_glove = np.array(xtrain_glove)
#xtest_glove = np.array(xtest_glove)

scores = []
#submission = pd.DataFrame.from_dict({'id': test['id']})
#for class_name in class_names:
 #   train_target = data_concatenated[class_name]
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_ANN(X_train, X_test, y_train, y_test):
    history = History()
    ### Initialising the ANN
    classifier = Sequential()
    #
    # Adding the input layer and the first hidden layer, input_dim = 11 11 nodes eisodou oi 11 steiles gia to train,output_dim=6 oi exodoi nodes (11+1)/2 empeirika
    classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu', input_dim = 201))
    classifier.add(Dropout(p = 0.5))
    # Adding the second hidden layer,
    classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.5))
    classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.5))
#    classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
#    classifier.add(Dropout(p = 0.5))
#    classifier.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
#    classifier.add(Dropout(p = 0.5))
#    classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
#    classifier.add(Dropout(p = 0.5))
#    classifier.add(Dense(output_dim = 2048, init = 'uniform', activation = 'relu'))
#    classifier.add(Dropout(p = 0.5))
    
    # Adding the output layer , output_dim = 1 ena node stin exodo
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier = classifier.fit(X_train, y_train,validation_data=(X_test,y_test), batch_size = 64, nb_epoch = 120,callbacks=[history])
    

    return classifier

def predict(classifier,X_test):
    # Predicting the Test set results- Making the predictions and evaluating the model
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    return y_pred

from sklearn.metrics import confusion_matrix,accuracy_score
from keras.models import load_model

## Making the Confusion Matrix
##cm = confusion_matrix(y_test, y_pred)

#
#def save_model(save_filepath):
#    classifier.save(save_filepath)

    
#classifier.summary()
def neural_print(title):
    from ann_visualizer.visualize import ann_viz;
    ann_viz(classifier, title=title)

#def load_my_model(save_filepath):
#    classifier = load_model(save_filepath)
#    return classifier


def Build_CV_ANN(X_train, X_test, y_train, y_test):
    # Evaluating the ANN
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from keras.models import Sequential
    from keras.layers import Dense
    
    def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu', input_dim = 201))
        classifier.add(Dropout(p = 0.1))
        # Adding the second hidden layer,
        classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.1))
        classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.1))
        classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.1))
        classifier.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.1))
        classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.1))
        #Adding the output layer , output_dim = 1 ena node stin exodo
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
        return classifier
    
    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 128, epochs = 120)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
    mean = accuracies.mean()
    variance = accuracies.std()
    print('CV Mean:',mean)
    print('CV Variance',variance)
    return classifier

def make_curve(history):
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
def plot_neural(classifier):
    from ann_visualizer.visualize import ann_viz;
    ann_viz(classifier, title="GLOVE_ANN")

#############  MAIN  #################
    
#save_filepath = '/home/kwstas/Desktop/Thesis_Python/Ml_Learning_models/Ann models saves/Model1/GLOVE/Ann_GLOVE_clf.h5'

#Build the ANN model
history = build_ANN(X_train, X_test, y_train, y_test)

#Build a CV=10 ANN
classifier = Build_CV_ANN(X_train, X_test, y_train, y_test)

#Load a compiled model
#classifier = load_my_model(save_filepath)

y_pred = predict(classifier,X_test)
print(accuracy_score(y_test, y_pred))

make_curve(history)
plot_neural(history)
#Save Keras Model
#save_model(save_filepath)