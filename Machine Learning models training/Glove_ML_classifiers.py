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
from nltk.stem.porter import PorterStemmer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

df = pd.read_csv('socialmedia_disaster_tweets.csv',delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
#df = df.loc[df.choose_one!="Can't Decide",:]
#df.reset_index(drop=True, inplace=True)
#df['choose_one'].replace(['Relevant','Not Relevant'],[1,0],inplace=True)
df.columns

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
    print(words)
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


df = pd.DataFrame(index=np.arange(0, data_concatenated.shape[0]), columns=[np.arange(0,50)])

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
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

### 4-Parameters Dictionary###
cl_params = {

    "DT": {
        "min_samples_split": range(3, 20,),
        "criterion" :["gini", "entropy"],
        "max_features" : ["log2"],
        "max_depth" : [5,6,7,8,9,10,11,12,13,14,15]
    },
    "LR": {
        "solver": ['newton-cg', 'sag','saga','lbfgs','liblinear'],
        "penalty" : [ "l2"],
        "dual" : [False],
        "C" :[0.01,0.03,0.05,0.07,0.09,1,1.5,2],
        "fit_intercept" : [True, False],
        "intercept_scaling" : [0.0001, 0.001,0.01,0.1,1,2],
        "max_iter" : [50,80,100,120,150],
        "tol" : [0.0001, 1e-1,1e-2,1e-3],
        "warm_start" : [True, False] 
    },
    "RF": {
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': range(10, 201, 20),       
        'criterion' : ['gini','entropy'],
        'bootstrap' : [True],
        'oob_score' : [False, True],
        "max_depth" : [10,20,30,40,50,60]
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
        'max_iter' : [50,100,150,200,250,300],
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
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

#### 5-Initialize the  models
abc = AdaBoostClassifier(algorithm="SAMME.R")
bc = BaggingClassifier(base_estimator=LogisticRegression(),random_state=7)
sgd = GradientBoostingClassifier( random_state=7)
rfc = RandomForestClassifier()
lrc = LogisticRegression()
nbc = GaussianNB()
dtc = DecisionTreeClassifier()
knc = KNeighborsClassifier()
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
import numpy as np

# Evaluate perfomance
def performance_eval(f_beta, acc):       
    # Fbeta score print
    print ("fbeta_score {}".format(f_beta))
    # Accuracy score print
    print ("accuracy_score {}".format(acc))
    
    
    
# Visualize training and validation curves
def performance_train_val_viz(model, X_train, y_train, scorer):    
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
  
    # Make an fbeta_score scoring object using make_scorer()
    scorer = make_scorer(fbeta_score, beta=1.0)
    
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
        grid_search = RandomizedSearchCV(clfs[learner], cl_params[learner], cv=10, scoring=scorer, random_state = 42,n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params[learner] = grid_search.best_params_
        print(best_params[learner])
        end = time.time() 

   
    
    # Calculate the training time
    results['train_time'] = end - start
    # Get the predictions on the X_test set,then get predictions on the X_train set using .predict()
    start = time.time() # Get start time
    predictions_test = grid_search.predict(X_test)
    predictions_train = grid_search.predict(X_train)
    end = time.time() # Get end time
    
    
    
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
    performance_train_val_viz(clfs[learner], X_train, y_train, scorer)

   

from sklearn.metrics import make_scorer, fbeta_score
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
import numpy as np 
import time

for k, v in clfs.items():
    print(k)
    trained_cl = train_predict(k, X_train, y_train, X_test, y_test)

performance(viz_results_train,viz_results_test)
