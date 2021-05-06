
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[ ]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def blight_model():
    
        #read in data
    train = pd.read_csv('train.csv', encoding='latin1')
    test = pd.read_csv('test.csv')
    addresses = pd.read_csv('addresses.csv')
    coordinates = pd.read_csv('latlons.csv')
    
    #merge coordinates to training and test
    train = pd.merge(train, pd.merge(addresses,coordinates, on='address'), on='ticket_id')
    test = pd.merge(test, pd.merge(addresses,coordinates, on='address'), on='ticket_id')

    #drop columns for tickets where person was found not responsible
    train.dropna(subset=['compliance'],inplace=True)

    #drop columns in train only
    train.drop(['payment_amount','payment_date','payment_status','balance_due','collection_status',
                'compliance_detail',
    #drop violator name (not generalizable) and address info columns since info is in lat, lon
    #keeping country of violator
                 'address','violator_name','violation_street_number', 'violation_street_name', 'violation_zip_code',
                 'mailing_address_str_number','mailing_address_str_name', 'city', 'state', 'zip_code',
                 'non_us_str_code',
    #             #drop other columns
                 'grafitti_status','inspector_name','ticket_issued_date','hearing_date', 'violation_description'], axis=1, inplace=True)

    #drop necessary columns from test to match train
    train_cols = list(train.columns.values)
    train_cols.remove('compliance')
    test = test[train_cols]

    train.dropna(inplace=True)

    #impute missing lat/lon values
    test['lat'] = test['lat'].fillna(test['lat'].mean())
    test['lon'] = test['lon'].fillna(test['lon'].mean())



    #CONVERT - categorical data to numerical for target and features
    labelencoder = LabelEncoder()
    cat_features = ['agency_name','country','violation_code','disposition']

    for i in cat_features:
        labelencoder.fit(train[i].append(test[i])) 
        train[i] = labelencoder.transform(train[i])
        test[i] = labelencoder.transform(test[i])

    #split out features and target from training set
    X = train.drop(['compliance'], axis=1)
    y = train['compliance']

    #split train and test from training set, not using ultimate test set
    X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)


    #fit a logistic regression
    #lr = LogisticRegression().fit(X_train, y_train)
    # lr_predicted = lr.predict(X_test)
    # confusion = confusion_matrix(y_test, lr_predicted)

    #use grid search to find the best parameters
    lr = LogisticRegression()
    grid_values = {'penalty': ['l1','l2']}

    # default metric to optimize over grid parameters: accuracy
    # grid_clf_acc = GridSearchCV(lr, param_grid = grid_values)
    # grid_clf_acc.fit(X_train, y_train)
    # y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 

    #print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
    #print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

    # alternative metric to optimize over grid parameters: AUC
    grid_clf_auc = GridSearchCV(lr, param_grid = grid_values, scoring = 'roc_auc')
    grid_clf_auc.fit(X_train, y_train)
    y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 

    #print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
    #print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    #print('Grid best score (AUC): ', grid_clf_auc.best_score_)

    answer = pd.DataFrame(grid_clf_auc.predict_proba(test)[:,1], test.ticket_id)
    
    return answer


# In[ ]:

blight_model()

