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

# Add nvcc path
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
    return data[0:1]

def load_lib(filename):
    """
    Load tvm lib and build executable graph
    Used to handle those hardware which do not support sklearn
    """
    lib = tvm.runtime.load_module(filename + ".tar")
    dev = tvm.device("llvm", 0)
    model = graph_executor.GraphModule(lib["default"](dev))
    return model

def run_graph(data, model, batch_size):
    """
    Run executable graph and get result
    """
    out = np.empty([data.shape[0]], dtype="float32")
    data = data.asnumpy()
    n_batch = data.shape[0] // batch_size
    load_time = 0
    exec_time = 0 
    store_time = 0
    for i in range(n_batch):
        start = i * batch_size
        end = (i + 1) * batch_size
        a = time.perf_counter()
        input_data = tvm.nd.array(data[start:end])
        model.set_input("data", input_data)
        b = time.perf_counter()
        model.run()
        c = time.perf_counter()
        out[start:end] = model.get_output(0).asnumpy().flatten()
        d = time.perf_counter()
        load_time = load_time + b - a
        exec_time = exec_time + c - b
        store_time = store_time + d - c
    print("load time:" + str(load_time))
    print("exec time:" + str(exec_time))
    print("store time:" + str(store_time))
    IO_time = load_time + store_time
    computation_time = exec_time 
    return out, IO_time, computation_time

def build_rewriting(sklearn_func, dataset, model_dir, data_dir, out_dir, batch_size=None):
    """
    build model into tvm lib fortmat
    fiting for tree-based models which can not execute on 32-bit raspberrypi
    """
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    if(batch_size == None):
        batch_size = data.shape[0]
    clf = load_model(model_dir, func_name, dataset)
    for target in target_list:
        model = build_model(clf, data.shape, sparse_replacing=False, dtype_converting=False, batch_size=batch_size, target=target)
        filename = os.path.join(out_dir, func_name + "_base_" + str(batch_size))
        model.save_model(filename)
        model = build_model(clf, data.shape, sparse_replacing=False, dtype_converting=True, batch_size=batch_size, target=target)
        filename = os.path.join(out_dir, func_name + "_dtype_" + str(batch_size))
        model.save_model(filename)
        model = build_model(clf, data.shape, sparse_replacing=True, dtype_converting=False, batch_size=batch_size, target=target)
        filename = os.path.join(out_dir, func_name + "_sparse_" + str(batch_size))
        model.save_model(filename)
        model = build_model(clf, data.shape, sparse_replacing=True, dtype_converting=True, batch_size=batch_size, target=target)
        filename = os.path.join(out_dir, func_name + "_both_" + str(batch_size))
        model.save_model(filename)
    return 0


def build_rasp(models, datasets, target_list, n_repeat, model_dir, data_dir, out_dir, batch_size):
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                build_rewriting(model, dataset, model_dir, data_dir, out_dir, batch_size)
    return 0

def run_rewriting(df, sklearn_func, dataset, model_dir, data_dir, batch_size):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    data = tvm.nd.array(data)
    results = [func_name, dataset]
    for target in target_list:
        results.append(target)
        filename = os.path.join(model_dir, func_name + "_base_" + str(batch_size))
        model = load_lib(filename)
        out, IO_time, computation_time = run_graph(data, model, batch_size)
        cmlcompiler_time = IO_time + computation_time
        results.append(cmlcompiler_time)
        filename = os.path.join(model_dir, func_name + "_dtype_" + str(batch_size))
        model = load_lib(filename)
        out, IO_time, computation_time = run_graph(data, model, batch_size)
        cmlcompiler_time = IO_time + computation_time
        results.append(cmlcompiler_time)
        filename = os.path.join(model_dir, func_name + "_sparse_" + str(batch_size))
        model = load_lib(filename)
        out, IO_time, computation_time = run_graph(data, model, batch_size)
        cmlcompiler_time = IO_time + computation_time
        results.append(cmlcompiler_time)
        filename = os.path.join(model_dir, func_name + "_both_" + str(batch_size))
        model = load_lib(filename)
        out, IO_time, computation_time = run_graph(data, model, batch_size)
        cmlcompiler_time = IO_time + computation_time
        results.append(cmlcompiler_time)
        df.loc[len(df)] = results
        results = [func_name, dataset]
    return df

def _breakdown(df, sklearn_func, dataset, model_dir, data_dir, batch_size):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    data = tvm.nd.array(data)
    results = [func_name, dataset]
    for target in target_list:
        results.append(target)
        filename = os.path.join(model_dir, func_name + "_base_" + str(batch_size))
        model = load_lib(filename)
        out, IO_time, computation_time = run_graph(data, model, batch_size)
        results.append(IO_time)
        results.append(computation_time)
        df.loc[len(df)] = results
        results = [func_name, dataset]
    return df


def run_rasp(models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size):
    columns = ["model", "dataset", "target", "base", "dtype", "sparse", "both"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = run_rewriting(df, model, dataset, model_dir, data_dir, batch_size)
    return df

def run_breakdown(models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size):
    columns = ["model", "dataset", "target", "IO", "computation"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _breakdown(df, model, dataset, model_dir, data_dir, batch_size)
    return df

def _elimination(df, sklearn_func, dataset, model_dir, data_dir, target_list, batch_size):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    clf = load_model(model_dir, func_name, dataset)
    number = 1
    check_flag = False
    results = [func_name, dataset] 
    for target in target_list:
        results.append(target)
        model = build_model(clf, data.shape, batch_size=batch_size, target=target, sparse_replacing=False, dtype_converting=False, elimination=False)
        load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, number, data, check_flag=check_flag)
        results.append(cmlcompiler_time)
        model = build_model(clf, data.shape, batch_size=batch_size, target=target, sparse_replacing=False, dtype_converting=False, elimination=True)
        load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, number, data, check_flag=check_flag)
        results.append(cmlcompiler_time)
        df.loc[len(df)] = results
        results = [func_name, dataset] 
    return df

def run_elimination(models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size):
    columns = ["model", "dataset", "target", "base", "elimination"]
    models = [SGDClassifier]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _elimination(df, model, dataset, model_dir, data_dir, target_list, batch_size)
    return df

models = [
        #Binarizer,
        #Normalizer,
        #MinMaxScaler,
        #RobustScaler,
        #LinearRegression, 
        #LogisticRegression, 
        #Preceptron,
        #SGDClassifier,
        DecisionTreeClassifier, 
        DecisionTreeRegressor, 
        RandomForestClassifier
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
#cross complication to build lib
target_list = ["llvm -model=bcm2711 -mtriple=armv8l-linux-gnueabihf -mattr=+neon -mcpu=cortex-a72"]
model_dir = "depth4"
#batchsize_list = [1721, 3442, 5163, 8605, 10326, 17210, 25815, 51630]

num_repeat = int(sys.argv[1])
savefile = sys.argv[2]
batchsize_list = [1721, 3442, 5163, 8605, 10326, 17210, 25815, 51630]
#for batch in batchsize_list:
#    df = test_rewriting(models, datasets, target_list, 1, model_dir, "test_datasets", batch)
#    print(df)
#df = test_rewriting(models, datasets, target_list, 1, model_dir, "test_datasets", 10326)
build_rasp(models, datasets, target_list, num_repeat, model_dir, "test_datasets", "rasp_model", 1)
#df = run_rasp(models, datasets, target_list, num_repeat, "rasp_model", "test_datasets", 1721)
#df = run_breakdown(models, datasets, target_list, num_repeat, "rasp_model", "test_datasets", 1721)
#df = run_elimination(models, datasets, target_list, num_repeat, "depth4", "test_datasets", None)
#df.to_csv(savefile, mode = "a", index=False)
#print(df)
