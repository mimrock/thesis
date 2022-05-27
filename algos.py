from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import VotingClassifier

import xgboost as xgb

'''
def get_algos():
    return [
        {"label": "XGB()", "fitter": xgb.XGBClassifier()},
        {"label": "XGB(mlogloss)", "fitter": xgb.XGBClassifier(eval_metric='mlogloss')},
        {"label": "XGB(evmetric=auc)", "fitter": xgb.XGBClassifier(eval_metric='auc')},
        {"label": "XGB(sp_w=8, evmetric=auc)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, eval_metric='auc')},
        {"label": "XGB(sp_w=8, evmetric=mlogloss)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, eval_metric='mlogloss')},
        {"label": "XGB(sp_w=8)", "fitter": xgb.XGBClassifier(scale_pos_weight=8)},
        {"label": "XGB(sp_w=9)", "fitter": xgb.XGBClassifier(scale_pos_weight=9, eval_metric='mlogloss')},
        {"label": "XGB(sp_w=9, evm=mlogloss)", "fitter": xgb.XGBClassifier(scale_pos_weight=9)},
        {"label": "XGB(sp_w=8, maxdepth=3)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, model__max_depth=3)},
        {"label": "XGB(sp_w=8, maxdepth=3, evm=mlogloss)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, model__max_depth=3, eval_metric='mlogloss')},
        {"label": "XGB(sp_w=8, maxdepth=20)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, model__max_depth=20)},
        {"label": "XGB(sp_w=8, maxdepth=20, evm=mlogloss)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, model__max_depth=20, eval_metric='mlogloss')},
        {"label": "XGB(maxdepth=20, evm=mlogloss)", "fitter": xgb.XGBClassifier(model__max_depth=20, eval_metric='mlogloss')},
        {"label": "XGB(sp_w=8, n_estimators=500)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, model__n_estimators=500)},
        {"label": "XGB(sp_w=8, n_estimators=500, evm=mlogloss)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, model__n_estimators=500, eval_metric='mlogloss')},
        {"label": "XGB(n_estimators=500, evm=mlogloss)", "fitter": xgb.XGBClassifier(model__n_estimators=500, eval_metric='mlogloss')},
        {"label": "GaussianNB", "fitter": GaussianNB()},
        #{"label": "ComplementNB(alpha=0.5)", "fitter": ComplementNB(alpha=0.5)},
        {"label": "LINEAR-SVC(class_weight=balanced,dual=false)", "fitter": svm.LinearSVC(dual=False, class_weight="balanced")},
        {"label": "xgb(code/antaresnyc/where-is-the-problem-auc-score-reached-99-7) nestim=1k", "fitter": xgb.XGBClassifier(max_depth = 8,
                      n_estimators = 1000,
                      reg_lambda = 1.2, reg_alpha = 1.2,
                      min_child_weight = 1,
                      objective = 'binary:logistic',
                      learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5, eval_metric = 'auc')},
        {"label": "xgb(code/antaresnyc/where-is-the-problem-auc-score-reached-99-7 + sp_w=8 nestim=1k)", "fitter": xgb.XGBClassifier(scale_pos_weight=8, max_depth = 8,
                                                                                                                  n_estimators = 1000,
                                                                                                                  reg_lambda = 1.2, reg_alpha = 1.2,
                                                                                                                  min_child_weight = 1,
                                                                                                                  objective = 'binary:logistic',
                                                                                                                  learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5, eval_metric = 'auc')}
        #{"label": "SVC(kernel=rbf)", "fitter": svm.SVC(kernel="rbf")}, # too slow, did not finish int 70 minutes
        #{"label": "NEW", "fitter": PassiveAggressiveClassifier()},
        #{"label": "RandomForest(class_weight=balanced_subsample,oob_score=true)", "fitter": RandomForestClassifier(class_weight="balanced_subsample", oob_score=False)},
        #{"label": "RandomForest()", "fitter": RandomForestClassifier()},
        #{"label": "RandomForest(cw=balanced)", "fitter": RandomForestClassifier(class_weight="balanced_subsample")},
        #{"label": "RandomForest(cw=balanced, nest=200)", "fitter": RandomForestClassifier(class_weight="balanced_subsample", n_estimators=200)},
        #{"label": "KNN(weights=distance)", "fitter": KNeighborsClassifier(n_neighbors=8, weights="distance", leaf_size=100)},
        #{"label": "SVC(linerar w/ prob)", "fitter": SklearnClassifier(SVC(kernel='linear',probability=True))}
        #{"label": "GradBoost(loss=exponential,subsample=0.75)", "fitter": GradientBoostingClassifier(loss="exponential", subsample=0.75)},
        #{"label": "SGD (class_weight=balanced,loss=modified_huber)", "fitter": SGDClassifier(class_weight="balanced", loss="modified_huber")},
        #{"label": "SGD (class_weight=balanced)", "fitter": SGDClassifier(class_weight="balanced")},
        #{"label": "GDBoost(md=4, n_est=372)", "fitter": GradientBoostingClassifier(max_depth=4, n_estimators=372)},
        #{"label": "GDBoost(md=2, n_est=372)", "fitter": GradientBoostingClassifier(max_depth=2, n_estimators=372)},
        #{"label": "GDBoost(lr=0.5)", "fitter": GradientBoostingClassifier(learning_rate=0.5)},
        #{"label": "GDBoost(lr=2.0)", "fitter": GradientBoostingClassifier(learning_rate=2.0)},
        #{"label": "GDBoost(loss=expon)", "fitter": GradientBoostingClassifier(loss="exponential")},
        #{"label": "ADABoost()", "fitter": AdaBoostClassifier()},
        #{"label": "ADABoost(lr=2)", "fitter": AdaBoostClassifier(learning_rate=2)},
        #{"label": "ADABoost(gnb)", "fitter": AdaBoostClassifier(base_estimator=GaussianNB())},
        #{"label": "ADABoost(knn)", "fitter": AdaBoostClassifier(base_estimator=KNeighborsClassifier(weights="distance"))}
        #{"label": "MLP()", "fitter": MLPRegressor()},
        #{"label": "MLP(solver=sgd)", "fitter": MLPRegressor(solver="sgd")},
        #{"label": "MLP(solver=lbfgs)", "fitter": MLPRegressor(solver="lbfgs")},
        #{"label": "svc rbf", "fitter": svm.SVC(class_weight="balanced")},
        #{"label": "svc sigmoid", "fitter": svm.SVC(kernel="sigmoid",  class_weight="balanced")},
        # {"label": "ADABoost(svc)", "fitter": AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight="balanced"))}
    ]'''

def get_algos():
    algos = [
        {"label": "GaussianNB", "fitter": GaussianNB()},
        #{"label": "ComplementNB(alpha=0.5)", "fitter": ComplementNB(alpha=0.5)},
        {"label": "LINEAR-SVC(class_weight=balanced)", "fitter": svm.LinearSVC(class_weight="balanced")},
        {"label": "RandomForest()", "fitter": RandomForestClassifier()},
        {"label": "RandomForest(cw=balanced)", "fitter": RandomForestClassifier(class_weight="balanced_subsample")},
        {"label": "RandomForest(cw=balanced, nest=500)", "fitter": RandomForestClassifier(class_weight="balanced_subsample", n_estimators=500)},
        #{"label": "SGD (class_weight=balanced,loss=modified_huber)", "fitter": SGDClassifier(class_weight="balanced", loss="modified_huber")},
        {"label": "SGD (class_weight=balanced)", "fitter": SGDClassifier(class_weight="balanced")},
        {"label": "GDBoost()", "fitter": GradientBoostingClassifier()},
        {"label": "GDBoost(md=4, n_est=372)", "fitter": GradientBoostingClassifier(max_depth=4, n_estimators=372)},
        #{"label": "GDBoost(md=2, n_est=372)", "fitter": GradientBoostingClassifier(max_depth=2, n_estimators=372)},
        #{"label": "GDBoost(lr=0.5)", "fitter": GradientBoostingClassifier(learning_rate=0.5)},
        #{"label": "GDBoost(lr=2.0)", "fitter": GradientBoostingClassifier(learning_rate=2.0)},
        #{"label": "ADABoost()", "fitter": AdaBoostClassifier()},
        #{"label": "ADABoost(lr=2)", "fitter": AdaBoostClassifier(learning_rate=2)},
        #{"label": "ADABoost(gnb)", "fitter": AdaBoostClassifier(base_estimator=GaussianNB())},
        {"label": "KNN()", "fitter": KNeighborsClassifier()},
        {"label": "KNN(weights=distance)", "fitter": KNeighborsClassifier(weights="distance")},
        {"label": "XGB()", "fitter": xgb.XGBClassifier()},
        {"label": "XGB(sp_w=2)", "fitter": xgb.XGBClassifier(scale_pos_weight=2)},
    ]

    '''algos.append({"label": "Voting(rf, xgb, sgf)", "fitter": VotingClassifier(estimators=[
                    ('xgb', algos[1]["fitter"]),
                    ('svc', algos[6]["fitter"]),
                    ('knn', algos[10]["fitter"]),
                    #('sgd', algos[7]["fitter"]),
            ], voting='soft')})'''
    return algos