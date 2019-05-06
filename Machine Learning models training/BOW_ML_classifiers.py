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
import pickle

print('The scikit-learn version is {}.'.format(sklearn.__version__))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_dataset():
    dataset = pd.read_csv('**path**/Dataset.csv',delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
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

def make_corpus(number_twt):    
    corpus = []
    for i in range(0,number_twt):
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
#class_hist(dataset)
#missing_values(dataset)
corpus = make_corpus(number_twt)
X,y= Bow_Split(corpus,dataset,max_features=500)
X_train, X_test, y_train, y_test = Test_Train_Split(X,y,test_size = 0.3)




# Save models in pickle and JSON files
def save_model_pickle(model,model_name):
    filename = 'Models/Model1/' +model_name+'classifier.joblib.pkl'
    _ = joblib.dump(model, filename, compress=9)
    print(filename)
     
     
     
### 4-Parameters Dictionary###
cl_params = {

    "DT": {
        "min_samples_split": range(3, 20,),
        "criterion" :["gini", "entropy"],
        "max_features" : ["log2"]
    },
    "LR": {
        "solver": ['newton-cg', 'sag','saga','lbfgs'],
        "penalty" : [ "l2"],
        "dual" : [False],
        "C" :[2.6,2.7,2.8,2.9,3.0],
        "fit_intercept" : [True, False],
        "intercept_scaling" : [0.0001, 0.001],
        "max_iter" : [1000],
        "tol" : [0.001, 1e-1,1e-2,1e-3],
        "warm_start" : [True, False] 
    },
    "RF": {
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': range(10, 201, 20),       
        'criterion' : ['gini','entropy'],
        'bootstrap' : [True],
        'oob_score' : [False, True]
    },
    "AdaBoost": {
        "n_estimators": range(2, 200)
    },
    "BgC": {
        "n_estimators": range(2, 200),
        'max_samples' : [0.05, 0.1, 0.2, 0.5],
    },
    "NB":{
            "var_smoothing":[1e-09,1e-10]
    },
    "MLP": {
        "activation": ['identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['lbfgs', 'sgd', 'adam'],
        'max_iter' : [1000],
        "tol" : [0.001, 1e-1,1e-2,1e-3],
        "hidden_layer_sizes" :[50,100,150,200],
        "batch_size":[100,120,140,160]
    } 
}
    
best_params = {

    "DT": {
            
        
    },
    "LR": {

    },
    "RF": {

    },
    "AdaBoost": {

    },
    "BgC": {

    },
    "NB":{

    },
    "MLP": {

    } 
}

viz_results_test = {
    "DT":{
        
        
    },
    "LR": {
        
        

    },
    "RF": {
        
        

    },
    "AdaBoost": {
        
        

    },
    "BgC": {
        
        

    },
    "NB":{
 
    },
    "MLP": {

    } 
            }
viz_results_train = {
    "DT":{

    },
    "LR": {


    },
    "RF": {


    },
    "AdaBoost": {
 

    },
    "BgC": {
 

    },
    "NB":{
  

    },
    "MLP": {

    } 
            }
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold,cross_val_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

#### 5-Initialize the  models
abc = AdaBoostClassifier(algorithm="SAMME.R")
bc = BaggingClassifier(base_estimator=LogisticRegression(),random_state=7)
rfc = RandomForestClassifier()
lrc = LogisticRegression()
nbc = GaussianNB()
dtc = DecisionTreeClassifier()
mlp = MLPClassifier()
    
clfs = {
    'LR': lrc,
    'DT': dtc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'NB' :  nbc, 
    'MLP' : mlp
}

# Construction of hyper parameters grid to tune
# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score, fbeta_score
import datetime
from time import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.model_selection import validation_curve,learning_curve
import json

# Evaluate perfomance
def performance_eval(f_beta, acc):       
    # Fbeta score print
    print ("fbeta_score {}".format(f_beta))
    # Accuracy score print
    print ("accuracy_score {}".format(acc))
    
    
    
# Visualize training and validation curves
def performance_train_val_viz(model, X_train, y_train, scorer):    
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, 
                                                            X_train, 
                                                            y_train,
                                                            cv=10,
                                                            scoring=scorer,
                                                            n_jobs=-1,
                                                            shuffle=True,
                                                            train_sizes=np.linspace(0.1, 1.0, 5),
                                                            random_state=42)

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def performance(train,test):
    
    N = 7
    trainMeans = (train["LR"], train["DT"], train["RF"], train["AdaBoost"], train["BgC"],train["NB"],train["MLP"])
    
    fig, ax = plt.subplots()
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35         # the width of the bars
    
    p1 = ax.bar(ind, trainMeans, width, color='r', bottom=0)
    testMeans = (test["LR"], test["DT"], test["RF"], test["AdaBoost"], test["BgC"],test["NB"],test["MLP"])
    p2 = ax.bar(ind + width, testMeans, width,color='y', bottom=0)
    
    ax.set_title('Performance')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('LR', 'DT', 'RF', 'Adaboost','BgC', 'NB','MLP'))
    
    ax.legend((p1[0], p2[0]), ('Train Score', 'Test Score'))
    ax.autoscale_view()
    
    plt.show()

def train_predict(learner, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    # Make an fbeta_score scoring object using make_scorer()
    #scorer = make_scorer(fbeta_score, beta=1.0)
    scorer ='accuracy'
    
    # Initialize a dictionary to save the results.
    results = {}
    start = time.time() # Get start time
    # Fit the learner to the training data using 
    if learner=="NB" :
        grid_search = GridSearchCV(clfs[learner], cl_params[learner], cv=10, scoring=scorer,n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params[learner] = grid_search.best_params_
        print(best_params[learner])
        end = time.time() #End time 
    else:
        grid_search = RandomizedSearchCV(clfs[learner], cl_params[learner], cv=10,n_jobs=-1, scoring=scorer, random_state = 42)
        grid_search.fit(X_train, y_train)
        best_params[learner] = grid_search.best_params_
        print(best_params[learner])
        end = time.time()
        
    # Save model
    print('Saving the model...')
    save_model_pickle(grid_search,learner)
    
    # Calculate the training time
    results['train_time'] = end - start
    # Get the predictions on the X_test set,then get predictions on the X_train set using .predict()
    start = time.time()
    predictions_test = grid_search.predict(X_test)
    predictions_train = grid_search.predict(X_train)
    end = time.time() 
    
    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on  training samples 
    results['acc_train'] = accuracy_score(y_train, predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on  training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train, predictions_train,beta = 1.0)
    viz_results_train[learner] = fbeta_score(y_train, predictions_train,beta = 1.0)
    viz_results_test[learner] = fbeta_score(y_test, predictions_test,beta = 1.0)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test,beta = 1.0)
    
    print("="*30)
    # Print performance results on train set
    print('Results on training set:')
    performance_eval(results['f_train'],results['acc_train'])
    
    # Print performance results on test set
    print('Results on test set:')
    performance_eval(results['f_test'],results['acc_test'])
    
    # Visualize learning (train, validation) curves
#    performance_train_val_viz(grid_search, X_train, y_train, scorer)
    

from sklearn.metrics import make_scorer, fbeta_score
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
import time

for k, v in clfs.items():
    print(k)
    trained_cl = train_predict(k, X_train, y_train, X_test, y_test)

#performance(viz_results_train,viz_results_test)
#scorer ='accuracy'


for k, v in clfs.items():
    print(k)
    clf2 = joblib.load('Models/Model1/' + k +'classifier.joblib.pkl')
    #performance_train_val_viz(clf2,X,y,scorer)
    clf2.predict(X_test)


