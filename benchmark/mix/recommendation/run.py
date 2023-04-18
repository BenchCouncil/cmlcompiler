import pandas as pd
import numpy as np
import pickle
import torch
import time
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import sys

test_data_path = "avazu/test_data.csv"
gbdt_model_path = "avazu/gbdt_model"
wide_deep_model_path = "avazu/wide_deep.pt"

def run_baseline(X):
    """
    Run baseline
    GBDT based on sklearn 
    WideAndDeep based on pytorch
    """
    gbdt_model = pickle.load(open(gbdt_model_path, 'rb'))
    wide_deep_model = torch.load(wide_deep_model_path)
    start = time.perf_counter()    
    X_feature = gbdt_model.transform([X])
    X_feature = torch.tensor(X_feature, dtype=torch.long)
    #print(X_feature)
    with torch.no_grad():
        y = wide_deep_model(X_feature)
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
    batch_size = 1
    lib_gbdt = tvm.runtime.load_module(hardware + "_gbdt.tar")
    model_gbdt = graph_executor.GraphModule(lib_gbdt["default"](dev))
    lib_wide_deep = tvm.runtime.load_module(hardware + "_wide_deep.tar")
    model_wide_deep = graph_executor.GraphModule(lib_wide_deep["default"](dev))   
    encoder_array = np.loadtxt('encoder_array', dtype=int).reshape(batch_size, -1)
    start = time.perf_counter()
    feature = np.zeros((batch_size, X.shape[0]))
    feature[:] = X
    model_gbdt.set_input("data", [X])
    model_gbdt.run()
    encoder_index = model_gbdt.get_output(0)
    encoder_index = encoder_index.asnumpy()
    encoder_array[0][encoder_index[0]] = 1
    X_feature = np.concatenate((feature, encoder_array), axis=-1)
    model_wide_deep.set_input("X_feature", X_feature)
    model_wide_deep.run()
    out = model_wide_deep.get_output(0)
    end = time.perf_counter()
    exec_time = end - start
    return exec_time

def run(hardware):
    single_sample = np.loadtxt(test_data_path, dtype=int, delimiter=",")
    columns = ["framework", "pipeline", "time"]
    df = pd.DataFrame(columns=columns)
    if(hardware != "pi"):
        result = ["baseline_"+hardware, "GBDT+Wide&Deep"]
        time_baseline = run_baseline(single_sample)
        result.append(time_baseline)
        df.loc[len(df)] = result
    result = ["cmlcompiler_"+hardware, "GBDT+Wide&Deep"]
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
