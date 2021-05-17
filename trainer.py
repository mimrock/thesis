#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score

from sklearn.naive_bayes import ComplementNB

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import VotingClassifier


import pandas as pd
from pandas_profiling import ProfileReport
import csv

import numpy as np

import logging

logging.basicConfig(format="%(asctime)-15s %(message)s", level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

USE_IGNORE = "ignore"
USE_TARGET = "target"
USE_FEATURE = "feature"

ENCODE_DEFAULT = "default"
ENCODE_ONE_HOT = "one_hot"

from sklearn.preprocessing import OneHotEncoder

import yaml

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

        print(plan)
        self.data_path = plan['data_file']
        self.mapping = plan['mapping']
        self.test_ratio = plan['test_ratio']

        self.default = {
            'use': 'feature',
            'encode': 'default'
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

    def load_data(self):
        '''with open(self.plan.data_path) as f:
            r = csv.reader(f)
            rows = []
            colnum = 0
            header = next(r)
            for row in r:
                rows.append(row)

        for field in header:
            if self.plan.fields[field]['use'] == USE_IGNORE:
                continue'''

        #@todo fix onehot encoder
        for field in list(self.data.head()):
            if self.plan[field]['use'] == USE_TARGET or self.plan[field]['use'] == USE_IGNORE:
                size = 0
            elif self.plan[field]['encode'] == ENCODE_ONE_HOT:
                self.encoders[field] = OneHotEncoder(sparse=False)
                vals = self.data[field].to_numpy()
                self.encoders[field].fit(vals.reshape(vals.shape[0], 1))
                size = self.encoders[field].categories_[0].shape[0]
            elif self.plan[field]['encode'] == ENCODE_DEFAULT:
                size = 1

            self.X_cols += size

        X = np.zeros((len(self.data.index), self.X_cols))
        y = np.zeros((len(self.data.index),))

        for i in range(0, len(self.data.index)):
            blah = X[i]
            X[i], y[i] = self.vectorize(self.data.iloc[i])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.plan.test_ratio, random_state=0)

        logging.debug("X_Train shape: %s", X_train.shape)

        algos = [
            {"label": "GaussianNB", "fitter": GaussianNB()},
            {"label": "ComplementNB(alpha=0.5)", "fitter": ComplementNB(alpha=0.5)},
            {"label": "LINEAR-SVC(class_weight=balanced,dual=false)", "fitter": svm.LinearSVC(class_weight="balanced", dual=False)},
            {"label": "RandomForest(class_weight=balanced_subsample,oob_score=true)", "fitter": RandomForestClassifier(class_weight="balanced_subsample", oob_score=True)},
            {"label": "KNN(weights=distance)", "fitter": KNeighborsClassifier(weights="distance")},
            #{"label": "GradBoost(loss=exponential,subsample=0.75)", "fitter": GradientBoostingClassifier(loss="exponential", subsample=0.75)},
            {"label": "SGD (class_weight=balanced,loss=modified_huber)", "fitter": SGDClassifier(class_weight="balanced", loss="modified_huber")}
        ]
            #{"label": "SVC(kernel=poly,cache_size=6000)", "fitter": svm.SVC(kernel="poly", cache_size=6000)}]

        algos.append({"label": "Voting(gnb, lsvc, rf, same params as above)", "fitter": VotingClassifier(estimators=[
                ('gnb', algos[0]["fitter"]),
                ('lsvc', algos[1]["fitter"]),
                ('rf', algos[3]["fitter"])
        ],voting='soft')})

        for algo in algos:
            logging.info("training: %s",algo["label"])
            m = algo["fitter"].fit(X_train, y_train)
            logging.info("predicting: %s", algo["label"])
            y_pred = m.predict(X_test)
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
            elif self.plan[field]['encode'] == ENCODE_ONE_HOT:
                # size = self.data[field].nunique()
                vals = np.array([[datarow[field]]])
                transformed = np.array(self.encoders[field].transform(vals))
                v[n:n+transformed.shape[1]] = transformed
                n += transformed.shape[1]
            elif self.plan[field]['encode'] == ENCODE_DEFAULT:
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
# @todo drop default as a field value. use should be: feature, ignore and target. ignore should be default.
# @todo original and one_hot for encoding features (better encoders if there's time)
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