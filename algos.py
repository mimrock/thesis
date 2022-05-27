from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# eXtreme Gradient Boosting
import xgboost as xgb


def get_algos():
    algos = [
        {"label": "GaussianNB", "fitter": GaussianNB()},
        {"label": "LINEAR-SVC(class_weight=balanced)", "fitter": svm.LinearSVC(class_weight="balanced")},
        {"label": "RandomForest()", "fitter": RandomForestClassifier()},
        {"label": "RandomForest(class_weight=balanced, n_estimators=500)", "fitter": RandomForestClassifier(class_weight="balanced_subsample", n_estimators=500)},
        {"label": "KNN()", "fitter": KNeighborsClassifier()},
        {"label": "KNN(weights=distance)", "fitter": KNeighborsClassifier(weights="distance")},
        {"label": "SGD (class_weight=balanced)", "fitter": SGDClassifier(class_weight="balanced")},
        {"label": "GDBoost()", "fitter": GradientBoostingClassifier()},
        {"label": "GDBoost(md=4, n_est=372)", "fitter": GradientBoostingClassifier(max_depth=4, n_estimators=372)},
        {"label": "XGB()", "fitter": xgb.XGBClassifier()},
        {"label": "XGB(spw=2)", "fitter": xgb.XGBClassifier(scale_pos_weight=2)},
    ]

    return algos
