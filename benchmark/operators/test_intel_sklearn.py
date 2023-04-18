from sklearn.preprocessing import Binarizer,LabelBinarizer,Normalizer,LabelEncoder
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
from sklearn.feature_selection import VarianceThreshold,SelectKBest,SelectPercentile
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,SVC,SVR
import numpy as np
import pandas as pd
from cmlcompiler.utils.supported_ops import tree_clf,ensemble_clf,preprocessing_op,feature_selectors
from benchmark.utils import bench_sklearn,bench_hb,bench_hb_all,bench_cmlcompiler,generate_dataset,check_data
import os
import pickle
import sys

def load_model(model_dir, func_name, dataset):
    #Load model
    filename = func_name + "_" + dataset + ".sav"
    filename = os.path.join(model_dir, filename)
    clf = pickle.load(open(filename, 'rb'))
    print(clf)
    return clf

def load_data(data_dir, dataset):
    #Load data
    data_name = os.path.join(data_dir, dataset + ".dat")
    data = pickle.load(open(data_name, 'rb'))
    return data

def _model(df, sklearn_func, dataset, model_dir, data_dir, number, batch_size, test_latency):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    if(test_latency == True):
        data = data[0:1]
    sk_time = 0
    if((sklearn_func in preprocessing_op) or (sklearn_func in feature_selectors)):
        clf = sklearn_func().fit(data)
    else:
        clf = load_model(model_dir, func_name, dataset)
    sk_time = bench_sklearn(clf, data, number=number, check_flag=False)    
    results = [func_name, dataset] 
    results.append(sk_time)
    df.loc[len(df)] = results
    return df

models = [
        Binarizer,
        Normalizer,
        MinMaxScaler,
        RobustScaler,
        LinearRegression, 
        LogisticRegression, 
        SGDClassifier,
        DecisionTreeClassifier, 
        DecisionTreeRegressor, 
        RandomForestClassifier,
        ExtraTreeClassifier,
        ExtraTreesClassifier,
        LinearSVC,
	    LinearSVR,
        ]

def test_framework(models, datasets, n_repeat, model_dir, data_dir, batch_size):
    columns = ["model", "dataset", "intel_sklearn"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _model(df, model, dataset, model_dir, data_dir, 1, batch_size, False)
    return df

savefile = sys.argv[1]
df = test_framework(models, ["year"], 5, "depth4", "test_datasets", None)
df.to_csv(savefile, mode = "a", index=False, header=True)

def test_latency(models, datasets, n_repeat, model_dir, data_dir, batch_size):
    columns = ["model", "dataset", "intel_sklearn"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _model(df, model, dataset, model_dir, data_dir, 1, batch_size, True)
    return df

savefile_latency = sys.argv[2]
df = test_latency(models, ["year"], 5, "depth4", "test_datasets", None)
df.to_csv(savefile_latency, mode = "a", index=False, header=True)
