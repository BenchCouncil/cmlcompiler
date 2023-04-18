import torch
import pickle
from sklearn.ensemble import RandomForestClassifier
import time
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from cmlcompiler.model import build_model
from benchmark.mix.radio_image.simpleDNN import simple_feature
import pandas as pd
import sys

forest_path = "forest"
simple_feature_path = "simple_feature.pt"

def run_baseline(X): 
    forest = pickle.load(open(forest_path, 'rb'))
    simple_feature = torch.load(simple_feature_path)
    start = time.perf_counter()
    with torch.no_grad():
        X_feature = simple_feature(X)
    y_pred = forest.predict(X_feature)
    end = time.perf_counter()
    exec_time = end - start
    return exec_time

def run_cmlcompiler(hardware, X):
    if(hardware == "cpu"):
        target = "llvm -mcpu=core-avx2"
    elif(hardware == "pi"):
        target = "llvm -model=bcm2711 -mtriple=armv8l-linux-gnueabihf -mattr=+neon -mcpu=cortex-a72"
    else:
        target = "llvm"
    dev = tvm.device(target, 0)
    lib_DNN = tvm.runtime.load_module(hardware + "_DNN.tar")
    model_DNN = graph_executor.GraphModule(lib_DNN["default"](dev))
    lib_forest = tvm.runtime.load_module(hardware + "_forest.tar")
    model_forest = graph_executor.GraphModule(lib_forest["default"](dev))   
    start = time.perf_counter()
    model_DNN.set_input("X", X)
    model_DNN.run()
    X_feature = model_DNN.get_output(0)
    model_forest.set_input("data", X_feature)
    model_forest.run()
    out = model_forest.get_output(0)
    end = time.perf_counter()
    exec_time = end - start
    return exec_time

def run(hardware):
    single_sample = torch.load("single_sample.pt")
    columns = ["framework", "pipeline", "time"]
    df = pd.DataFrame(columns=columns)
    if(hardware != "pi"):
        result = ["baseline_"+hardware, "DNN+RF"]
        time_baseline = run_baseline(single_sample)
        result.append(time_baseline)
        df.loc[len(df)] = result
    result = ["cmlcompiler_"+hardware, "DNN+RF"]
    time_cmlcompiler = run_cmlcompiler(hardware, single_sample)    
    result.append(time_cmlcompiler)
    df.loc[len(df)] = result
    return df

df = run(sys.argv[1])
if(sys.argv[2] == "False"):
    save_header = False
else:
    save_header = True
savefile = sys.argv[3]
df.to_csv(savefile, mode = "a", index=False, header=save_header)