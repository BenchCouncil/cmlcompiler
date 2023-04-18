"""testing models"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from tvm.contrib import graph_executor
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
import time
import os 
import pickle
from cmlcompiler.model import build_model,tune_model,load_tune,tune_log_name
import sys

os.environ["TVM_BACKTRACE"] = "1"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

def load_model(model_dir, func_name, dataset):
    # Load model
    filename = func_name + "_" + dataset + ".sav"
    filename = os.path.join(model_dir, filename)
    clf = pickle.load(open(filename, 'rb'))
    return clf

def convert_clf_classes_to_int(clf):
    """
    clf.classes_ in sklearn is float
    convert to int for hummingbird
    """
    if(hasattr(clf, "classes_")):
        clf.classes_ = [int(i) for i in clf.classes_]
        if(type(clf) in (tree_clf + ensemble_clf)):
            clf.classes_ = np.array(clf.classes_)
    return clf

def load_data(data_dir, dataset):
    # Load data
    data_name = os.path.join(data_dir, dataset + ".dat")
    data = pickle.load(open(data_name, 'rb'))
    return data

def _model(df, breakdown, sklearn_func, dataset, model_dir, data_dir, number, target_list, batch_size, check_flag):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    if((sklearn_func in preprocessing_op) or (sklearn_func in feature_selectors)):
        if(sklearn_func in [LabelEncoder]):
            data = data[:,0]
        clf = sklearn_func().fit(data)
    else:
        clf = load_model(model_dir, func_name, dataset)
    if(check_flag == True):
        sk_time, sklearn_out = bench_sklearn(clf, data, number=number, check_flag=check_flag)
    else:
        try:
            sk_time = bench_sklearn(clf, data, number=number, check_flag=check_flag)
        except Exception as e:
            sk_time = -1
            pass
    clf = convert_clf_classes_to_int(clf)
    input_data = np.array(data)
    results = [func_name, dataset] 
    results.append(sk_time)
    for target in target_list:
        model = build_model(clf, data.shape, batch_size=batch_size, target=target, sparse_replacing=False, dtype_converting=False, elimination=True)
        if(check_flag == True):
            load_time, exec_time, store_time, take_time, cmlcompiler_time, cmlcompiler_out = bench_cmlcompiler(model, number, data, check_flag=check_flag)
            if(check_data(sklearn_out, cmlcompiler_out) == False):
                cmlcompiler_time = -1
        else:
            load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, number, data, check_flag=check_flag)
        IO_time = load_time + store_time
        computation_time = exec_time + take_time
        results.append(cmlcompiler_time)
        breakdown_results = [func_name, dataset, target, IO_time, computation_time]
        breakdown.loc[len(breakdown)] = breakdown_results
    df.loc[len(df)] = results
    return df, breakdown

def test_framework(models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size, check_flag=False):
    """
    test the results of different frameworks
    """
    columns = ["model", "dataset", "sklearn"] + target_list
    df = pd.DataFrame(columns=columns)
    breakdown_columns = ["model", "dataset", "target", "IO", "computation"]
    breakdown = pd.DataFrame(columns=breakdown_columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df, breakdown = _model(df, breakdown, model, dataset, model_dir, data_dir, 1, target_list, batch_size, check_flag)
    return df, breakdown

models = [
        #Binarizer,
        #Normalizer,
        #MinMaxScaler,
        #RobustScaler,
        #LinearRegression, 
        #LogisticRegression, 
        #Perceptron,
        #SGDClassifier,
        DecisionTreeClassifier, 
        #DecisionTreeRegressor, 
        #RandomForestClassifier,
        #ExtraTreeClassifier,
        #LinearSVR,
        #LinearSVC
        ]

datasets = [
    #"fraud", 
    "year",
    #"higgs", 
    #"epsilon", 
    #"airline" 
    ]
target_list = ["llvm"]
model_dir = "depth4"
#model_dir = "test_models"
#batchsize_list = [1721, 3442, 5163, 8605, 10326, 17210, 25815, 51630]
#df = test_batchsize(models, datasets, target_list, 1, "depth4", "test_datasets", batchsize_list, True)
#df.to_csv("result_batchsize.csv", mode = "a", index=False)

n_model = int(sys.argv[1])
savefile = sys.argv[2]
breakdown_file = sys.argv[3]
if(sys.argv[4] == "False"):
    save_header = False
else:
    save_header = True
test_models = [models[n_model]]
df, breakdown = test_framework(test_models, datasets, target_list, 1, model_dir, "test_datasets", batch_size=None, check_flag=False)
df.to_csv(savefile, mode = "a", index=False, header=save_header)
breakdown.to_csv(breakdown_file, mode = "a", index=False, header=save_header)

