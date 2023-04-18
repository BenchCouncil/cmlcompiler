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
        sk_time = bench_sklearn(clf, data, number=number, check_flag=check_flag)
    clf = convert_clf_classes_to_int(clf)
    """
    if(check_flag == False):
        hb_time = bench_hb_all(clf, data, number=number, check_flag=check_flag)
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
    """
    hb_time = [-1, -1, -1, -1]
    results = [func_name, dataset] 
    results.append(sk_time)
    for t in hb_time:
        results.append(t)
    for target in target_list:
        model = build_model(clf, data.shape, batch_size=batch_size, target=target, sparse_replacing=True, dtype_converting=False, elimination=True)
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
    df.loc[len(df)] = results
    return df, breakdown

def _batch(df, sklearn_func, dataset, model_dir, data_dir, batchsize_list, target_list, largest=True):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    clf = load_model(model_dir, func_name, dataset)
    results = [func_name, dataset]
    if(largest == True):
        batchsize_list.append(data.shape[0])
    for target in target_list:
        results.append(target)
        for bs in batchsize_list:
            model = build_model(clf, data.shape, batch_size=bs, target=target)
            cmlcompiler_time = bench_cmlcompiler(model, 1, data, check_flag=False)
            results.append(cmlcompiler_time)
        df.loc[len(df)] = results
        results = [func_name, dataset]
    if(largest == True):
        batchsize_list.pop()
    return df

def _rewriting(df, sklearn_func, dataset, model_dir, data_dir, batch_size=None):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    if(batch_size == None):
        batch_size = data.shape[0]
    clf = load_model(model_dir, func_name, dataset)
    results = [func_name, dataset]
    for target in target_list:
        results.append(target)
        model = build_model(clf, data.shape, sparse_replacing=False, dtype_converting=False, batch_size=batch_size, target=target)
        load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, 1, data, check_flag=False)
        results.append(cmlcompiler_time)
        model = build_model(clf, data.shape, sparse_replacing=False, dtype_converting=True, batch_size=batch_size, target=target)
        load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, 1, data, check_flag=False)
        results.append(cmlcompiler_time)
        model = build_model(clf, data.shape, sparse_replacing=True, dtype_converting=False, batch_size=batch_size, target=target)
        load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, 1, data, check_flag=False)
        results.append(cmlcompiler_time)
        model = build_model(clf, data.shape, sparse_replacing=True, dtype_converting=True, batch_size=batch_size, target=target)
        load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, 1, data, check_flag=False)
        results.append(cmlcompiler_time)
        df.loc[len(df)] = results
        results = [func_name, dataset]
    return df

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

def test_batchsize(models, datasets, target_list, n_repeat, model_dir, data_dir, batchsize_list, largest):
    """
    test the influence of batch size
    """
    columns = ["model", "dataset", "target"] + batchsize_list
    if(largest == True):
        columns = columns + ["max"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _batch(df, model, dataset, model_dir, data_dir, batchsize_list, target_list, largest)
    return df

def test_rewriting(models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size):
    """
    test the influence of graph rewriting
    "sparse" means only using sparse replacing
    "dtype" means only using dtype converting
    "both" means using both
    """
    columns = ["model", "dataset", "target", "base", "dtype", "sparse", "both"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _rewriting(df, model, dataset, model_dir, data_dir, batch_size)
    return df


def _test(df, sklearn_func, dataset, model_dir, data_dir, batch_size=None):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    print(type(data))
    if(batch_size == None):
        batch_size = data.shape[0]
    clf = load_model(model_dir, func_name, dataset)
    results = [func_name, dataset]
    for target in target_list:
        results.append(target)
        model = build_model(clf, data.shape, sparse_replacing=False, dtype_converting=False, batch_size=batch_size, target=target)
        cmlcompiler_time = bench_cmlcompiler(model, 1, data, check_flag=False)
        results.append(cmlcompiler_time)
        df.loc[len(df)] = results
        results = [func_name, dataset]
    return df

def test(models, datasets, target_list, n_repeat, model_dir, data_dir):
    columns = ["model", "dataset", "target", "result"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _test(df, model, dataset, model_dir, data_dir)
    return df

def _tune(df, sklearn_func, dataset, model_dir, data_dir, sparse_replacing, dtype_converting, batch_size=None, load=False):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    data = tvm.nd.array(data)
    print(data.shape)
    if(batch_size == None):
        batch_size = data.shape[0]
    clf = load_model(model_dir, func_name, dataset)
    results = [func_name, dataset]
    for target in target_list:
        log_file = tune_log_name(func_name, target, sparse_replacing, dtype_converting)
        log_file = model_dir + "/" + "tuning_log/" + dataset + "/" + log_file
        results.append(target)
        if(load == True):
            model = load_tune(clf, data.shape, log_file=log_file, sparse_replacing=False, dtype_converting=False, batch_size=batch_size, target=target)
        else:
            model = tune_model(clf, data.shape, log_file=log_file, n_trials=20000, sparse_replacing=sparse_replacing, dtype_converting=dtype_converting, batch_size=batch_size, target=target)
        cmlcompiler_time = bench_cmlcompiler(model, 1, data, check_flag=False)
        results.append(cmlcompiler_time)
        df.loc[len(df)] = results
        results = [func_name, dataset]
    return df

def test_tuning(models, datauets, target_list, n_repeat, model_dir, data_dir, load=False):
    columns = ["model", "dataset", "target", "result"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _tune(df, model, dataset, model_dir, data_dir, False, False, load=load)
    return df

models = [
        #Binarizer,
        #Normalizer,
        #MinMaxScaler,
        #MaxAbsScaler,
        #StandardScaler,
        #RobustScaler,
        #LinearRegression, 
        #LogisticRegression, 
        #DecisionTreeClassifier, 
        #DecisionTreeRegressor, 
        RandomForestClassifier,
        #RandomForestRegressor,
        #LinearSVR,
        #LinearSVC,
        #SVR
        ]
datasets = [
    #"fraud", 
    "year",
    #"higgs", 
    #"epsilon", 
    #"airline" 
    ]
#target_list = ["llvm -mcpu=core-avx2", "llvm"]
#target_list = ["llvm -mcpu=core-avx2", "llvm", "cuda"]
target_list = ["cuda"]
model_dir = "depth4"
#model_dir = "test_models"
#batchsize_list = [1721, 3442, 5163, 8605, 10326, 17210, 25815, 51630]
#df = test_batchsize(models, datasets, target_list, 1, "depth4", "test_datasets", batchsize_list, True)
#print(df)
#df.to_csv("result_batchsize.csv", mode = "a", index=False)

n_model = int(sys.argv[1])
savefile = sys.argv[2]
breakdown_file = sys.argv[3]
if(sys.argv[4] == "False"):
    save_header = False
else:
    save_header = True
test_models = [models[n_model]]
#df, breakdown = test_framework(test_models, datasets, target_list, 1, model_dir, "test_datasets", batch_size=None, check_flag=True)
#df.to_csv(savefile, mode = "a", index=False, header=save_header)
#breakdown.to_csv(breakdown_file, mode = "a", index=False, header=save_header)
"""
batchsize_list = [1721, 3442, 5163, 8605, 10326, 17210, 25815, 51630]
for batch in batchsize_list:
    df = test_rewriting(models, datasets, target_list, 1, model_dir, "test_datasets", batch)
    print(df)
"""
#df = test_rewriting(models, datasets, target_list, 1, model_dir, "test_datasets", 10326)
#df = test_rewriting(models, datasets, target_list, 5, model_dir, "test_datasets", 1721)
#df = test_rewriting(models, datasets, target_list, 1, model_dir, "test_datasets", None)
#df.to_csv("result_rewriting.csv", mode = "a", index=False)
#print(df)
df = test_tuning(models, datasets, target_list, 1, model_dir, "test_datasets", True)
print(df)
