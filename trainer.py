from sklearn.model_selection import train_test_split

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

import argparse

from plan import Plan
from encoders import MyNormalizer
from result import Result, HtmlResult

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
                logging.debug("setting up one_hot preprocessor for field: %s", field)
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
                logging.debug("setting up label preprocessor for field: %s", field)
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
            self.X[i], self.y[i] = self.vectorize(self.data.iloc[i])
            logging.debug("X[i]: %s y[i]: %s", self.X[i], self.y[i])

    def train(self):

        logging.debug("first data row: %s", self.data.iloc[0])
        logging.debug("second data row: %s", self.data.iloc[1])
        logging.info("X first row: %s", self.X[0])
        logging.info("X second row: %s", self.X[1])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)

        logging.debug("X_Train shape: %s", self.X_train.shape)
        logging.info("X train first few rows: %s", self.X_train[0:5])
        logging.info("y train first few rows: %s", self.y_train[0:5])

        logging.info("X test first few rows: %s", self.X_test[0:20])
        logging.info("y test first few rows: %s", self.y_test[0:20])



    def validate(self):

        results = []
        for algo in self.algos:
            logging.info("training: %s",algo["label"])
            m = algo["fitter"].fit(self.X_train, self.y_train)
            logging.info("predicting: %s", algo["label"])
            y_pred = m.predict(self.X_test)
            results.append(Result(self.y_test, y_pred, algo["label"]))
        return results



    def vectorize(self, datarow):
            v = np.zeros((self.X_cols,))
            target = np.zeros((1,))
            n = 0
            for field in self.plan:
                if self.plan[field]['use'] == USE_IGNORE:
                    continue
                elif self.plan[field]['use'] == USE_TARGET:
                    # @todo encoding
                    logging.debug('target field: %s val: %s', field, datarow[field])
                    target[0] = datarow[field]
                elif self.plan[field]['preprocess'] == PREPROCESS_ONE_HOT:
                    vals = np.array([[datarow[field]]])
                    transformed = np.array(self.encoders[field].transform(vals))
                    v[n:n+transformed.shape[1]] = transformed
                    logging.debug('1hot field: %s val: %s encoded: %s place(%s:%s)', field, datarow[field], transformed, n, n+transformed.shape[1])
                    n += transformed.shape[1]
                elif self.plan[field]['preprocess'] == PREPROCESS_NORMALIZE:
                    vals = np.array([[datarow[field]]])
                    v[n:n+1] = self.encoders[field].transform_single(vals[0])
                    logging.debug('normalized field: %s val: %s encoded: %s place(%s)', field, datarow[field], v[n:n+1], n)
                    n += 1
                elif self.plan[field]['preprocess'] == PREPROCESS_LABEL:
                    vals = np.array([[datarow[field]]])
                    transformed = self.encoders[field].transform(vals.reshape(1,))
                    v[n:n+1] = transformed
                    logging.debug('label field: %s val: %s encoded: %s place(%s)', field, datarow[field], transformed, n)
                    n += 1
                elif self.plan[field]['preprocess'] == PREPROCESS_SCALE:
                    vals = np.array([[datarow[field]]])
                    transformed = self.encoders[field].transform(vals)
                    v[n:n+1] = transformed
                    n += 1
                elif self.plan[field]['preprocess'] == PREPROCESS_ORIGINAL:
                    v[n] = datarow[field]
                    logging.debug('orig field: %s val: %s place(%s)', field, datarow[field], n)
                    n += 1

            return v, target


parser = argparse.ArgumentParser()

parser.add_argument('--plan',
                    help='Path to the yaml file containing the plan.', required=True)

parser.add_argument('--htmlresults',
                    help='Path to the html file where the results will be generated.'
                         'If not set, the results will only be displayed as logs')

args = parser.parse_args()

logging.info("Loading data")
#p = Plan('data/kaggle/health-insurance-cross-sell-prediction/plan.yaml')
#p = Plan('data/kaggle/titanic/plan.yaml')
p = Plan(args.plan)
m = Manager(p)
m.load_data()
logging.info("Training models")
m.train()
logging.info("Validating models")
results = m.validate()

if args.htmlresults is not None:
    hr = HtmlResult(results)
    hr.write_html(args.htmlresults)
else:
    for r in results:
        r.as_logs()


# @todo for sunday
    # x html export using the css
    # x split classes to separate files
    # - maybe an automatical voting one?
    # - create a separate file that generates the pandas html report (highlight highest values if there is time)
    # - LATER: Create a new repo, clean up, clear unnecessary files, only keep necessary, clear comments.
    # - LATER: If a miracle happens, add more functionality, e.g. cross-validation support

    # WRITE: about formula of normalization, voting comittes, titanic dataset, argparser, html export, string format
