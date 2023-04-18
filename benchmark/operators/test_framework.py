"""testing models"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
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

# Add nvcc path
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
os.environ["TVM_BACKTRACE"] = "1"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
np.set_printoptions(threshold=1000)

def load_model(model_dir, func_name, dataset):
    # Load model
    filename = func_name + "_" + dataset + ".sav"
    filename = os.path.join(model_dir, filename)
    clf = pickle.load(open(filename, 'rb'))
    print(clf)
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
    sk_time = 0
    if((sklearn_func in preprocessing_op) or (sklearn_func in feature_selectors)):
        if(sklearn_func in [LabelEncoder]):
            data = data[:,0]
        clf = sklearn_func().fit(data)
    else:
        clf = load_model(model_dir, func_name, dataset)
    if(check_flag == True):
        sk_time, sklearn_out = bench_sklearn(clf, data, number=number, check_flag=check_flag)
    else:
        sk_time = bench_sklearn(clf, data, number=number, check_flag=check_flag)
    clf = convert_clf_classes_to_int(clf)
    if func_name not in ["ExtraTreeRegressor", "ExtraTreeClassifier"]:
        if(check_flag == False):
            if func_name in ["ExtraTreesClassifier"]:
                hb_tvm = -1
                hb_tvm_gpu = -1
                hb_py = bench_hb(clf, data, number=number, check_flag=check_flag, backend="pytorch")
                hb_py_gpu = bench_hb(clf, data, number=number, check_flag=check_flag, backend="pytorch", target="cuda")
            else:
                hb_py, hb_py_gpu, hb_tvm, hb_tvm_gpu = bench_hb_all(clf, data, number=number, check_flag=check_flag)
        else:
            hb_py, hb_py_gpu, hb_tvm, hb_tvm_gpu, out_hb_py, out_hb_py_gpu, out_hb_tvm, out_hb_tvm_gpu = bench_hb_all(clf, data, number=number, check_flag=check_flag)
            if(check_data(sklearn_out, out_hb_py) == False):
                hb_py = -1
            if(check_data(sklearn_out, out_hb_py_gpu) == False):
                hb_py_gpu = -1
            if(check_data(sklearn_out, out_hb_tvm) == False):
                hb_tvm = -1
            if(check_data(sklearn_out, out_hb_tvm_gpu) == False):
                hb_tvm_gpu = -1
        hb_time = [hb_py, hb_py_gpu, hb_tvm, hb_tvm_gpu]
    else:
        hb_time = [-1, -1, -1, -1]
    results = [func_name, dataset] 
    results.append(sk_time)
    for t in hb_time:
        results.append(t)
    if func_name not in ["ExtraTreesClassifier"]:
        for target in target_list:
            model = build_model(clf, data.shape, batch_size=batch_size, target=target, sparse_replacing=False, dtype_converting=False, elimination=False)
            if(check_flag == True):
                load_time, exec_time, store_time, take_time, cmlcompiler_time, cmlcompiler_out = bench_cmlcompiler(model, number, data, check_flag=check_flag)
                if(check_data(sklearn_out, cmlcompiler_out) == False):
                    cmlcompiler_time = -1
                """
                for i in range(sklearn_out.shape[0]):
                    if(sklearn_out[i] != cmlcompiler_out[i]):
                        print(i)
                        print(sklearn_out[i])
                        print(cmlcompiler_out[i])
                """
            else:
                load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, number, data, check_flag=check_flag)
            IO_time = load_time + store_time
            computation_time = exec_time + take_time
            results.append(cmlcompiler_time)
            breakdown_results = [func_name, dataset, target, IO_time, computation_time]
            breakdown.loc[len(breakdown)] = breakdown_results
    else:
        for target in target_list:
            results.append(float("inf"))
            breakdown_results = [func_name, dataset, target, float("inf"), float("inf")]
            breakdown.loc[len(breakdown)] = breakdown_results
    df.loc[len(df)] = results
    return df, breakdown

def test_framework(models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size, check_flag=False):
    """
    test the results of different frameworks
    """
    columns = ["model", "dataset", "sklearn", "hb_torch_cpu", "hb_torch_gpu", "hb_tvm_cpu", "hb_tvm_gpu"] + target_list
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
        #SGDClassifier,
        DecisionTreeClassifier, 
        #DecisionTreeRegressor, 
        #RandomForestClassifier,
        #ExtraTreesRegressor,
        #ExtraTreeClassifier,
        #ExtraTreesClassifier,
	    #LinearSVR,
        #LinearSVC,
        ]

datasets = [
    #"fraud", 
    "year",
    #"higgs", 
    #"epsilon",
    #"airline" 
    ]
#target_list = ["llvm -mcpu=core-avx2", "llvm"]
target_list = ["llvm -mcpu=core-avx2", "llvm", "cuda"]
model_dir = "depth4"
#model_dir = "test_models"

n_model = int(sys.argv[1])
savefile = sys.argv[2]
breakdown_file = sys.argv[3]
if(sys.argv[4] == "False"):
    save_header = False
else:
    save_header = True
test_models = [models[n_model]]
n_repeat = 1
df, breakdown = test_framework(test_models, datasets, target_list, n_repeat, model_dir, "test_datasets", batch_size=None, check_flag=False)
df.to_csv(savefile, mode = "a", index=False, header=save_header)
breakdown.to_csv(breakdown_file, mode = "a", index=False, header=save_header)
