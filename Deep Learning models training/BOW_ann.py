# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk 
import sklearn 
from sklearn.preprocessing import Imputer

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import History,EarlyStopping,ModelCheckpoint
#import sys
#sys.setrecursionlimit(1000)

Tweets_number = 79187
def read_dataset():
    dataset = pd.read_csv('Clean_Disasters_T_79187_.csv',delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
    return dataset
   
corpus = []

def make_corpus():    
    corpus = []
    for i in range(0,Tweets_number):
        corpus.append(dataset.text[i])
    return corpus

def Bow_Split(corpus,dataset,max_features): #### 2-Bag of words model
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    #Count Vecorizer
    cv = CountVectorizer(max_features = (max_features))
    X = cv.fit_transform(corpus).toarray() 
    
    ####Tf-Idf Vectorizer
    #tf = TfidfVectorizer(max_features=(50))
    #X = tf.fit_transform(corpus).toarray()
    
    #Split Dataset to X and y
    y = dataset.loc[: ,'choose_one'].values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    return X,y


def Test_Train_Split(X,y,test_size): #Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42,shuffle=True,stratify=dataset['choose_one'].values)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test



def build_ANN(X_train, X_test, y_train, y_test):
    history = History()
#    history = model.fit(X_train, y_train, batch_size=150, validation_data=(X_test, y_test),
#                    epochs=10, verbose=1,callbacks=[history])
    ### Initialising the ANN
    classifier = Sequential()
    #
    # Adding the input layer and the first hidden layer, input_dim = 11 11 nodes eisodou oi 11 steiles gia to train,output_dim=6 oi exodoi nodes (11+1)/2 empeirika
    classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu', input_dim = 500))
    classifier.add(Dropout(p = 0.5))
    # Adding the second hidden layer,
    classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.5))
    classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.5))
    #classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
    #classifier.add(Dropout(p = 0.5))
    #classifier.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
    #classifier.add(Dropout(p = 0.5))
    #classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
    #classifier.add(Dropout(p = 0.5))
    #classifier.add(Dense(output_dim = 2048, init = 'uniform', activation = 'relu'))
    #classifier.add(Dropout(p = 0.5))
    
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


def save_model(save_filepath):
    classifier.save(save_filepath)

    
#classifier.summary()
def neural_print(title):
    from ann_visualizer.visualize import ann_viz;
    ann_viz(classifier, title=title)

def load_my_model(save_filepath):
    classifier = load_model(save_filepath)
    return classifier


def Build_CV_ANN(X_train, X_test, y_train, y_test):
    # Evaluating the ANN
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from keras.models import Sequential
    from keras.layers import Dense
    history = History()
    
    def build_classifier():
    

        classifier = Sequential()
        classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu', input_dim = 500))
        classifier.add(Dropout(p = 0.1))
        # Adding the second hidden layer,
        classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.1))
        classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.1))
#        classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
#        classifier.add(Dropout(p = 0.1))
#        classifier.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
#        classifier.add(Dropout(p = 0.1))
#        classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
#        classifier.add(Dropout(p = 0.1))
        #Adding the output layer , output_dim = 1 ena node stin exodo
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
        return classifier
    
    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 64, epochs = 120,callbacks=[history])
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


######MAIN#########
save_filepath = '/home/kwstas/Desktop/Thesis_Python/ALL_models/Models/Ann models saves/Model1/BOW/Ann_BOW_clf.h5'
#REad dataset
dataset = read_dataset()

#Make the corpus
corpus = make_corpus()

#Split X,y and test,train
X,y= Bow_Split(corpus,dataset,max_features=500)
X_train, X_test, y_train, y_test = Test_Train_Split(X,y,test_size = 0.3)



#Build the ANN model
history = build_ANN(X_train, X_test, y_train, y_test)

#Build a CV=10 ANN
#classifier = Build_CV_ANN(X_train, X_test, y_train, y_test)

#Load a compiled model
classifier = load_my_model(save_filepath)



#y_pred = predict(classifier,X_test)
#print(accuracy_score(y_test, y_pred))



#MAke plots
make_curve(history)

#Save Keras Model
#save_model(save_filepath)
