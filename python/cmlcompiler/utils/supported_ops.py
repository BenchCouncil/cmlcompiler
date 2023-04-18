"""
Supported operators
"""
from sklearn.preprocessing import Binarizer,LabelBinarizer,Normalizer,LabelEncoder,MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
from sklearn.feature_selection import SelectKBest,VarianceThreshold
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,SVC,SVR

# Preprocessing operatros
preprocessing_op = [
        Binarizer,
        LabelBinarizer,
        Normalizer,
        LabelEncoder,
        MaxAbsScaler,
        MinMaxScaler,
        StandardScaler,
        RobustScaler
        ]

# Feature Selectors
feature_selectors = [
        SelectKBest,
        VarianceThreshold
        ]

# Linear-based classifier
linear_clf = [
        LogisticRegression,
        LogisticRegressionCV,
        Perceptron,
        RidgeClassifier,
        RidgeClassifierCV,
        SGDClassifier,
        LinearSVC
        ]

# Linear-based regressor
linear_reg = [
        LinearRegression,
        Ridge,
        RidgeCV,
        SGDRegressor,
        LinearSVR
        ]

# Linear-based models
linear_ops = linear_clf + linear_reg

# Tree-based classifier
tree_clf = [
        DecisionTreeClassifier,
        ExtraTreeClassifier
        ]

# Tree-based regressor
tree_reg = [
        DecisionTreeRegressor,
        ExtraTreeRegressor
        ]

# Tree-based models
tree_ops = tree_clf + tree_reg

# Ensemble Classifier
ensemble_clf = [
        RandomForestClassifier,
        ExtraTreesClassifier
        ]

# Ensemble Regressor
ensemble_reg = [
        RandomForestRegressor,
        ExtraTreesRegressor
        ]

# Ensemble models
ensemble_ops = ensemble_clf + ensemble_reg

#SVM Classifer
svm_clf = [
        SVC,
        NuSVC
        ]

# SVM Regressor
svm_reg = [
        SVR,
        NuSVR
        ]

# SVM
svm_ops = svm_clf + svm_reg

# All Classifiers
clf_ops = tree_clf + linear_clf + ensemble_clf + svm_clf

# All Regressors
reg_ops = tree_reg + linear_reg + ensemble_reg + svm_reg

# All operators
ops = clf_ops + reg_ops
