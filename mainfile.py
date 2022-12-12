""" importing main libraries"""
import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging# have to work on this

import warnings
warnings.filterwarnings('ignore')

""" Importing machinelearning libraries"""
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBRFClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import export_graphviz
from xgboost import XGBClassifier
import lightgbm

from sklearn.metrics import f1_score


from imblearn.over_sampling import RandomOverSampler

""" Setting up a logger"""
logging.basicConfig(filename=  'logfile.log', #name of the log file
                    filemode='w', #file mode used when opening the log file
                    level = logging.DEBUG)

#creating a logger object
logger = logging.getLogger()


#setting the threshold of the logger to Debug
""" importing the data to be used"""

train = pd.read_csv('train.use.csv')
test = pd.read_csv('test.csv')


""" Merging the datasets based on equal columns"""

Mdata = pd.concat([train, test], axis=0)




# print('..........."Merged dataset".......')



#
class df (pd.DataFrame):
    #i am inheriting directly from Dataframe and adding methods to it.
    def get_shape (self):
        output = Mdata.shape
        return output

    def get_dtype (self):
        output = Mdata.dtypes
        return output

    def get_missing_values (self):
        output = Mdata.isna().sum()
        return output

    def get_head (self):
        output = Mdata.head
        return output


Mdata = df(Mdata)
# print('....."This is the number of columns and rows".......''\n')
# print(Mdata.shape,'\n')
logging.debug(Mdata.shape)
# print('....."This is the first five".......''\n')
# print(Mdata.head(),'\n')
logging.debug(Mdata.head)
# print('....."This is the number of columns and rows".......''\n')
# print(Mdata.get_dtype(),'\n')
logging.debug(Mdata.get_dtype)
# print('....."This is the sum of missing values".......''\n')
# print(Mdata.get_missing_values(),'\n'
logging.debug(Mdata.get_missing_values)
#

""" This class is created to check Data preprocessing """

# class Preprocessing (df):
#
#     def info (self):
#         output = f' {self.dummies()}'
#         return output
#
#     def duplicate(self):# Checking if the data is duplicated or not
#         try:
#             if Mdata.duplicated == True:
#                 raise ValueError
#         except ValueError:
#             print('data duplicated')
#         else:
#             print('data is not duplicated')
#             return

#Here we define the X(features) and Y (features) might take out
x = Mdata.drop (columns=['policy_id', 'is_claim', 'area_cluster',
                         'make', 'model', 'fuel_type', 'airbags', 'cylinder'], axis = 1)
y = Mdata['is_claim']

# print(Mdata.columns)

#assigning dummies to the X (features variables)
X = pd.get_dummies(x, drop_first=True)
# print( X)

imb = Mdata['is_claim'].value_counts() / Mdata.shape[0] * 100

# print(imb)

# data is imbalanced I would have to balance it, else
# it makes the classification difficult"""

#using randomoversampler to handle imbalanced data
""" This is used to randomly duplicate examples in the minority
class so as to add them to the training dataset our sampling stragegy is =0.8
and to ensure that this is the same set run every time, I would set
random_state = 23"""


oversampler = RandomOverSampler(random_state=23, sampling_strategy=0.8)

X_res, y_res = oversampler.fit_resample(X, y)


# print(f'This is the current shape after oversampling {X_res.shape} and {y_res.shape}', '\n')



""" AFter balancing the data, we scale it.
Scaling the data: this is done to make the distance
between the feature of the data generalized so that we compare them"""


scaler=StandardScaler ()
X_res = scaler.fit_transform(X_res)

"""Checking to see if the resampled data is different
from the origial data shape"""

def counting (a,b):
    output = Counter (a), Counter (b)
    print (a, b)

# print('counting the data to see if there is a difference', '\n')
logger.debug(f'{counting(a= y, b=y_res)}')
# """ this is the data processing: Test-train splitting: This allows us to create the model
# on a portion of the data and test it on another portion"""
#

# class Processing (df):
#
#     def __init__(self):
#         print(self)
#
#     def __str__(self):
#         output = ()
#         return output


# print('We are splitting the data after balancing it...............', '\n')


X_train,X_test,y_train,y_test=train_test_split (X_res,y_res,test_size = 0.2, random_state = 1)






print('Data has been split')

"""tuning the hyper parameters of randomforest classifier"""

clf =RandomForestClassifier(n_estimators=100,
                           criterion= 'gini',
                           max_depth =None,
                           min_samples_leaf = 1,
                           min_samples_split= 5,
                           random_state=1)

""" This fits the randomforest to the training model (cross validation)"""

clf.fit(X_train, y_train)

logger.debug(clf.fit(X_train, y_train))

# ploting the randomforest on the first train
from tempfile import mkdtemp
savedir = mkdtemp()
import os
filename = os.path.join(savedir, 'test.joblib')
filename2 = os.path.join(savedir, 'test.joblib')
import joblib
joblib.dump(clf, filename, compress=3)

logger.debug(joblib.load(filename))
print(joblib.load(filename))

logger.debug(len(clf.estimators_))
print('we have this number of estimators')
print(len(clf.estimators_))

"""training set performance """
train_pred = clf.predict(X_train)
train_accuracy = f1_score(y_train, train_pred)

joblib.dump(clf, filename, compress=3)

fn = X_train.feature_names
estimators = clf.estimators_[2]
dot_data = export_graphviz(estimators, out_file='tree.dot',
                feature_names= fn,
                class_names= True,
                rounded= True, proportion=False,
                precision=2, filled=True)

dot_data.savefig('imagename.png')

joblib.dump(dot_data, filename2, compress=3)

# test_pred = clf.predict(X_test)
# test_accuracy = f1_score(y_test, test_pred)



logger.debug('Accuracy for Training set is')
logger.debug(100*train_accuracy)
print('......................................')
print('Accuracy for Testing set is')
# print(100*test_accuracy)


"""creating the final submission file"""


# submission = pd.DataFrame()
# submission['policy_id'] = policy_id
# submission['is_claim2'] = y_pred
# submission.to_csv('final.csv', index=False)


#
#
#

# print(Preprocessing.duplicate(Mdata))
# print(Preprocessing.XnY (Mdata))
# print(Preprocessing.dummies(Mdat))
# print(Preprocessing.info)
# print(Preprocessing.imb(Mdata))
# print('this data is imbalanced')
# print(Preprocessing.oversampler)


""" AFter balancing the data, we scale it.
Scaling the data: this is done to make the distance
between the feature of the data generalized so that we compare them"""
