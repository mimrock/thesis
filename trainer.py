#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

from sklearn.ensemble import VotingClassifier

import pandas as pd
from pandas_profiling import ProfileReport
import csv

import numpy as np

import logging

from algos import get_algos

logging.basicConfig(format="%(asctime)-15s %(message)s", level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

USE_IGNORE = "ignore"
USE_TARGET = "target"
USE_FEATURE = "feature"

PREPROCESS_ORIGINAL = "original"  # keep original data
PREPROCESS_ONE_HOT = "one_hot"  # one hot encoder
PREPROCESS_NORMALIZE = "normalize"  # normalize data between 0 and 1
PREPROCESS_LABEL = "label"  # transform labels to numbers
PREPROCESS_SCALE = "scale"  # transform labels to numbers

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

import yaml

class MyNormalizer:
    def __init__(self):
        pass

    def fit(self, array):
        self.min = array.min()
        self.max = array.max()

    def transform_single(self, val):
        if val < self.min or val > self.max:
            raise ValueError("Out of bound value")

        z = val - self.min
        return z / (self.max - self.min)


class Plan:
    def __init__(self, plan_file):
        # path to the plan file
        self.plan_file = plan_file

        with open(plan_file, 'r') as f:
            try:
                plan = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)

        self.data_path = plan['data_file']
        self.mapping = plan['mapping']
        self.test_ratio = plan['test_ratio']

        self.default = {
            'use': 'ignore',
            'preprocess': 'original'
        }

    #@todo fix, user_input gets overwritten.
    def apply(self, fields):
        plan = {}
        for field in fields:
            plan[field] = self.default.copy()
            if field in self.mapping:
                for key in self.mapping[field]:
                    plan[field][key] = self.mapping[field][key]

        return plan


class Manager:
    def __init__(self, plan):
        self.user_plan = plan
        self.data = pd.read_csv(self.user_plan.data_path)
        self.plan = self.user_plan.apply(list(self.data.head()))
        self.X_cols = 0
        self.encoders = {}
        self.algos = get_algos()

    def load_data(self):

        #@todo test
        #@todo do not vectorize line by line (this breaks Normalizer, and maybe other preprocessors, probably against best practices, and also slower)
        # also makes standardization impossible.

        # intialize preprocessors
        for field in list(self.data.head()):
            if self.plan[field]['use'] == USE_TARGET or self.plan[field]['use'] == USE_IGNORE:
                size = 0
            elif self.plan[field]['preprocess'] == PREPROCESS_ONE_HOT:
                self.encoders[field] = OneHotEncoder(sparse=False)
                vals = self.data[field].to_numpy()
                self.encoders[field].fit(vals.reshape(vals.shape[0], 1))
                size = self.encoders[field].categories_[0].shape[0]
            elif self.plan[field]['preprocess'] == PREPROCESS_NORMALIZE:
                self.encoders[field] = MyNormalizer()
                vals = self.data[field].to_numpy()
                self.encoders[field].fit(vals.reshape(vals.shape[0], 1))
                size = 1
            elif self.plan[field]['preprocess'] == PREPROCESS_LABEL:
                self.encoders[field] = LabelEncoder()
                vals = self.data[field].to_numpy()
                self.encoders[field].fit(vals.reshape(vals.shape[0], 1))
                size = 1
            elif self.plan[field]['preprocess'] == PREPROCESS_SCALE:
                self.encoders[field] = RobustScaler()
                vals = self.data[field].to_numpy()
                self.encoders[field].fit(vals.reshape(vals.shape[0], 1))
                size = 1
            elif self.plan[field]['preprocess'] == PREPROCESS_ORIGINAL:
                size = 1
            else:
                raise RuntimeError("Invalid preprocess type: {}".format(self.plan[field]['preprocess']))

            self.X_cols += size

        self.X = np.zeros((len(self.data.index), self.X_cols))
        self.y = np.zeros((len(self.data.index),))

        for i in range(0, len(self.data.index)):
            #blah = self.X[i]
            self.X[i], self.y[i] = self.vectorize(self.data.iloc[i])

    def train(self):

        logging.debug("first data row: %s", self.data.iloc[0])
        logging.debug("X first row: %s", X[0])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        logging.debug("X_Train shape: %s", self.X_train.shape)
        logging.debug("X first row: %s", self.X_train[0])

        self.algos.append({"label": "Voting(gnb, cnb, rf, sgd, gdm same params as above)", "fitter": VotingClassifier(estimators=[
                ('gnb', self.algos[0]["fitter"]),
                #('cnb', algos[1]["fitter"]),
                ('rf', self.algos[3]["fitter"]),
                ('sgd', self.algos[4]["fitter"]),
        ], voting='soft')})

    def validate(self):

        for algo in self.algos:
            logging.info("training: %s",algo["label"])
            m = algo["fitter"].fit(self.X_train, self.y_train)
            logging.info("predicting: %s", algo["label"])
            y_pred = m.predict(self.X_test)
            print("{}: y_pred shape: {} Recall: {} Precision: {}".format(algo["label"], y_pred.shape, recall_score(y_test, y_pred), precision_score(y_test, y_pred)))

    def vectorize(self, datarow):
        # print(datarow['Age'])
        v = np.zeros((self.X_cols,))
        target = np.zeros((1,))
        n = 0
        for field in self.plan:
            if self.plan[field]['use'] == USE_IGNORE:
                continue
            elif self.plan[field]['use'] == USE_TARGET:
                # @todo encoding
                target[0] = datarow[field]
            elif self.plan[field]['preprocess'] == PREPROCESS_ONE_HOT:
                vals = np.array([[datarow[field]]])
                transformed = np.array(self.encoders[field].transform(vals))
                v[n:n+transformed.shape[1]] = transformed
                n += transformed.shape[1]
            elif self.plan[field]['preprocess'] == PREPROCESS_NORMALIZE:
                vals = np.array([[datarow[field]]])
                v[n:n+1] = self.encoders[field].transform_single(vals[0])
                n += 1
            elif self.plan[field]['preprocess'] == PREPROCESS_LABEL:
                vals = np.array([[datarow[field]]])
                transformed = self.encoders[field].transform(vals.reshape(1,))
                v[n:n+1] = transformed
                n += 1
            elif self.plan[field]['preprocess'] == PREPROCESS_SCALE:
                vals = np.array([[datarow[field]]])
                transformed = self.encoders[field].transform(vals)
                v[n:n+1] = transformed
                n += 1
            elif self.plan[field]['preprocess'] == PREPROCESS_ORIGINAL:
                v[n] = datarow[field]
                n += 1

        return v, target



'''
@todo 
- check the expected shape for X,y
- what shapes to use for the single line vectors
- predict
- tests'''




'''print("field:", field, "cardinality:", data[field].nunique())
if self.plan.fields[field]['encode'] == ENCODE_ONE_HOT:
    enc = OneHotEncoder()
    enc.fit(np.array(data[field]))
    print("field categories:", enc.categories)'''

logging.info("Starting")
p = Plan('data/kaggle/health-insurance-cross-sell-prediction/plan.yaml')
m = Manager(p)
m.load_data()


# @todo argparse command line option for generating report and running a plan
# @todo create a requirements.txt
# @todo result should go to stdout, logs to stderr

'''df = pd.read_csv('data/kaggle/health-insurance-cross-sell-prediction/train.csv')
pd.options.display.max_columns = 10
print(df.describe())

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("your_report.html")

from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
svc = SVC(kernel="linear")
y_pred = svc.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))'''