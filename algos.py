from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPRegressor

def get_algos():
    return [
        {"label": "GaussianNB", "fitter": GaussianNB()},
        #{"label": "ComplementNB(alpha=0.5)", "fitter": ComplementNB(alpha=0.5)},
        {"label": "LINEAR-SVC(class_weight=balanced,dual=false)", "fitter": svm.LinearSVC(dual=False, class_weight="balanced")},
        #{"label": "NEW", "fitter": PassiveAggressiveClassifier()},
        {"label": "RandomForest(class_weight=balanced_subsample,oob_score=true)", "fitter": RandomForestClassifier(class_weight="balanced_subsample", oob_score=False)},
        {"label": "KNN(weights=distance)", "fitter": KNeighborsClassifier(n_neighbors=8, weights="distance", leaf_size=100)},
        #{"label": "SVC(linerar w/ prob)", "fitter": SklearnClassifier(SVC(kernel='linear',probability=True))}
        #{"label": "GradBoost(loss=exponential,subsample=0.75)", "fitter": GradientBoostingClassifier(loss="exponential", subsample=0.75)},
        {"label": "SGD (class_weight=balanced,loss=modified_huber)", "fitter": SGDClassifier(class_weight="balanced", loss="modified_huber")},
        {"label": "GDBoost()", "fitter": GradientBoostingClassifier(n_estimators=500)},
        {"label": "ADABoost()", "fitter": AdaBoostClassifier()},
        {"label": "ADABoost(lr=0.2)", "fitter": AdaBoostClassifier(learning_rate=0.2)},
        {"label": "ADABoost(gnb)", "fitter": AdaBoostClassifier(base_estimator=GaussianNB())},
        {"label": "ADABoost(knn)", "fitter": AdaBoostClassifier(base_estimator=KNeighborsClassifier(weights="distance"))}
        #{"label": "MLP()", "fitter": MLPRegressor()},
        #{"label": "MLP(solver=sgd)", "fitter": MLPRegressor(solver="sgd")},
        #{"label": "MLP(solver=lbfgs)", "fitter": MLPRegressor(solver="lbfgs")},
        #{"label": "svc rbf", "fitter": svm.SVC(class_weight="balanced")},
        #{"label": "svc sigmoid", "fitter": svm.SVC(kernel="sigmoid",  class_weight="balanced")},
        # {"label": "ADABoost(svc)", "fitter": AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight="balanced"))}
    ]