import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
import transformers
import pickle
import warnings
import time
from cmlcompiler.model import build_model
from benchmark.utils import bench_cmlcompiler
import os
import json
import sys

warnings.filterwarnings('ignore')
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
os.environ["TVM_BACKTRACE"] = "1"

def run_baseline(df, test_padded, test_mask):
    pretrained_weights = "prajjwal1/bert-tiny"
    input_ids = torch.tensor(test_padded, dtype=torch.long)
    attention_mask = torch.tensor(test_mask)
    model = transformers.AutoModel.from_pretrained(pretrained_weights, return_dict=False)
    clf = pickle.load(open("lr_clf.sav", "rb"))
    start = time.perf_counter()
    with torch.no_grad():
        bert_features = model(input_ids, attention_mask=attention_mask)[0].numpy()[:, 0, :]
    out = clf.predict(bert_features)
    end = time.perf_counter()
    results = ["baseline_cpu"]
    results.append("Bert+LR")
    results.append(end - start)
    df.loc[len(df)] = results
    return df

def run_cmlcompiler(df, test_padded, test_mask, hardware):
    if(hardware == "cpu"):
        target = "llvm -mcpu=core-avx2"
        results = ["cmlcompiler_cpu"]
    elif(hardware == "pi"):
        target = "llvm"
        results = ["cmlcompiler_pi"]
    dev = tvm.device(target, 0)
    input_ids = tvm.nd.array(test_padded)
    attention_mask = tvm.nd.array(test_mask)
    lib_bert = tvm.runtime.load_module(hardware + "_bert.tar")
    model_bert = graph_executor.GraphModule(lib_bert["default"](dev))
    lib_take = tvm.runtime.load_module(hardware + "_take.tar")
    model_take = graph_executor.GraphModule(lib_take["default"](dev))
    lib_tree = tvm.runtime.load_module(hardware + "_lr.tar")
    model_tree = graph_executor.GraphModule(lib_tree["default"](dev))
    
    start = time.perf_counter()
    model_bert.set_input("input_ids", input_ids)
    model_bert.set_input("attention_mask", attention_mask)
    model_bert.run()
    inter_data = model_bert.get_output(0)
    model_take.set_input("data", inter_data)
    model_take.run()
    inter_data = model_take.get_output(0)
    model_tree.set_input("data", inter_data)
    model_tree.run()
    out = model_tree.get_output(0)
    end = time.perf_counter()
    results.append("Bert+LR")
    results.append(end - start)
    df.loc[len(df)] = results
    return df

def run(hardware):
    test_padded = np.loadtxt("test_padded.csv", delimiter=",")[0:1]
    test_mask = np.loadtxt("test_mask.csv", delimiter=",")[0:1]
    columns = ["framework", "pipeline", "time"]
    df = pd.DataFrame(columns=columns)
    if(hardware == "cpu"):
        df = run_baseline(df, test_padded, test_mask)
        df = run_cmlcompiler(df, test_padded, test_mask, hardware)
    else:
        df = run_cmlcompiler(df, test_padded, test_mask, hardware)
    return df
    

df = run(sys.argv[1])
if(sys.argv[2] == "False"):
    save_header = False
else:
    save_header = True
savefile = sys.argv[3]
df.to_csv(savefile, mode = "a", index=False, header=save_header)
