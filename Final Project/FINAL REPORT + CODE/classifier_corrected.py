#!/usr/bin/env python

from __future__ import division

import re
import numpy as np
from sklearn import metrics, cross_validation, linear_model, preprocessing, ensemble
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import pandas as pd
from xgboost.sklearn import XGBClassifier
import XGBoostClassifier as xg
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
from sklearn.feature_extraction import DictVectorizer


rcParams['figure.figsize'] = 12, 4


SEED = 42  # always use a seed for randomized procedures


def pandasConvertData(filename):
    Data = pd.read_csv(filename, sep = ",", skiprows=1, names=['ACTION','RESOURCE','MGR_ID','ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_DEPTNAME','ROLE_TITLE','ROLE_FAMILY_DESC','ROLE_FAMILY','ROLE_CODE'], header=None)
    #Grabs salary column for use as .target
    DataSet = pd.read_csv(filename, sep = ",", skiprows=1, names=['ACTION','RESOURCE','MGR_ID','ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_DEPTNAME','ROLE_TITLE','ROLE_FAMILY_DESC','ROLE_FAMILY','ROLE_CODE'], header=None)

    label = Data['ACTION'] 
    #Deletes once done
    del Data['ACTION']
    del Data['ROLE_CODE']
    return label, Data, DataSet          



def load_data(filename, use_labels=True):


    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open(filename), delimiter=',',
                      usecols=range(1,9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open(filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):
    #Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def modelTraining(X, y, X_test, models):
    for item in models:
        item.fit(X, y)
        preds = item.predict_proba(X_test)[:, 1]
        m = re.search('(?:(?!-->).)*', str(item))
        save_results(preds, m.group(0) + ".csv")




def modelfit(alg, dtrain, predictors, dtest, dataSetTrain, dataSetTest, outputFileName, useTrainCV=False, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['ACTION'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    

    #Fit the algorithm on the data
    if outputFileName == "testXGB":
         alg.fit(dtrain[predictors], dtrain['ACTION'],eval_metric='auc')
    else:
         alg.fit(dtrain[predictors], dtrain['ACTION'])
        
    #Predict training set:
    dtrain_predprob = alg.predict_proba(dtest[predictors])[:,1]
    save_results(dtrain_predprob, outputFileName + ".csv")

def encode_onehot(df, cols):
 
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def main():
  #  """
  #  Fit models and make predictions.
  #  We'll use one-hot encoding to transform our categorical features
  #  into binary features.
  #  y and X will be numpy array objects.
  #   """
    models = []
    model1 = linear_model.LogisticRegression(C=3,random_state=SEED,n_jobs=-1)  # the classifier we'll use
    model2 = linear_model.LogisticRegressionCV(random_state=SEED, n_jobs=-1)
    #model3 = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED)
    model4 = ensemble.RandomForestClassifier(random_state = SEED)
    model5 = ensemble.ExtraTreesClassifier(random_state = SEED)
    modelXGB=xg.XGBoostClassifier(num_round=1000 ,nthread=25,  eta=0.12, gamma=0.01,max_depth=12, min_child_weight=0.01, subsample=0.6, 
                                   colsample_bytree=0.7,objective='binary:logistic',seed=1) 
    modelXGB2 = XGBClassifier(learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)

    models.append(model1)
    models.append(model2)
    #models.append(model3)
    models.append(model4)
    models.append(model5)
    models.append(modelXGB)
    models.append(modelXGB2)




    # === load data in memory === #
    print "loading data"
    y, X = load_data('../Original Files/train.csv')
    y_test, X_test = load_data('../Original Files/test.csv', use_labels=False)                

    # === one-hot encoding === #
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)

    #import data in pandas format
    labels, dataSetTrain, train = pandasConvertData('../Original Files/train.csv')
    labels2, dataSetTest, test = pandasConvertData('../Original Files/test.csv')

    #label predictor rows
    target = 'ACTION'
    dropCol = 'ROLE_CODE'
    predictors = [x for x in train.columns if x not in [target,dropCol]]

    #Encode pandas data set
    encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
    encoder.fit(dataSetTrain,dataSetTest)
    dataSetTrain = encoder.transform(dataSetTrain)  # Returns a sparse matrix (see numpy.sparse)
    dataSetTest = encoder.transform(dataSetTest)

    #2 Different methods below. First one takes all models in models[] and predicts probabilities, second one takes algorithms and uses pandas to manipulate. Metrics are also found in the second method
    modelTraining(X, y, X_test, models)
    #modelfit(xgb1, train, predictors, test, dataSetTrain, dataSetTest, "OH YEAH")



if __name__ == '__main__':
    main()
