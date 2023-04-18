from cmlcompiler.algorithms.linear import logistic_regression,logistic_regression_cv,ridge_classifier,ridge_classifier_cv,sgd_classifier,perceptron,linear_regression,ridge,ridge_cv,sgd_regressor
from cmlcompiler.algorithms.tree import decision_tree_classifier,decision_tree_regressor,extra_tree_classifier,extra_tree_regressor
from cmlcompiler.algorithms.forest import random_forest_classifier,random_forest_regressor,extra_trees_classifier,extra_trees_regressor
from cmlcompiler.algorithms.svm import linear_svc,linear_svr
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.svm import LinearSVC,LinearSVR

def func_name_parser(model):
    """
    Input sklearn model
    Return corresponding func
    """
    model_name = type(model)
    func_map = {
            LogisticRegression:logistic_regression,
            LogisticRegressionCV:logistic_regression_cv,
            Perceptron:perceptron,
            RidgeClassifier:ridge_classifier,
            RidgeClassifierCV:ridge_classifier_cv,
            SGDClassifier:sgd_classifier,
            LinearRegression:linear_regression,
            Ridge:ridge,
            RidgeCV:ridge_cv,
            SGDRegressor:sgd_regressor,
            DecisionTreeClassifier:decision_tree_classifier,
            DecisionTreeRegressor:decision_tree_regressor,
            ExtraTreeClassifier:extra_tree_classifier,
            ExtraTreeRegressor:extra_tree_regressor,
            RandomForestClassifier:random_forest_classifier,
            RandomForestRegressor:random_forest_regressor,
            ExtraTreesClassifier:extra_trees_classifier,
            ExtraTreesRegressor:extra_trees_regressor,
            LinearSVC:linear_svc,
            LinearSVR:linear_svr
            }
    return func_map[model_name]
