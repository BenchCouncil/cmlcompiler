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

warnings.filterwarnings('ignore')
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
os.environ["TVM_BACKTRACE"] = "1"

pretrained_weights = "prajjwal1/bert-tiny"
model = transformers.AutoModel.from_pretrained(pretrained_weights, return_dict=False)
padded = np.loadtxt("test_padded.csv", delimiter=",")[0:1]
mask = np.loadtxt("test_mask.csv", delimiter=",")[0:1]
#clf = pickle.load(open("tree_clf.sav", "rb"))
clf = pickle.load(open("lr_clf.sav", "rb"))

target = ["llvm -mcpu=core-avx2", "llvm -model=bcm2711 -mtriple=armv8l-linux-gnueabihf -mattr=+neon -mcpu=cortex-a72"]

def compile_to_lib(target, padded, mask, model, clf, hardware):
    dev = tvm.device(str(target), 0)
    input_ids = torch.tensor(padded, dtype=torch.long)
    attention_mask = torch.tensor(mask)
    scripted_model = torch.jit.trace(model, [input_ids, attention_mask], strict=False)
    shape_list = [("input_ids", input_ids.shape), ("attention_mask", attention_mask.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    lib.export_library(hardware + "_bert.tar")
    tvm_model = graph_executor.GraphModule(lib["default"](dev))
    input_ids = tvm.nd.array(input_ids)
    attention_mask = tvm.nd.array(attention_mask)
    tvm_model.set_input("input_ids", input_ids)
    tvm_model.set_input("attention_mask", attention_mask)
    tvm_model.run()
    inter_data = tvm_model.get_output(0)
    
    data_shape1 = inter_data.shape
    data = relay.var("data", shape=data_shape1, dtype="float32")
    out = relay.take(data, relay.const(0, "int32"), axis=1)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([data], out)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target)
    lib.export_library(hardware + "_take.tar")
    tvm_model = graph_executor.GraphModule(lib["default"](dev))
    tvm_model.set_input("data", inter_data)
    tvm_model.run()
    inter_data = tvm_model.get_output(0)

    data_shape2 = inter_data.shape
    model = build_model(clf, data_shape2, batch_size=1, target=target, sparse_replacing=False, dtype_converting=False, elimination=True)
    model.save_model(hardware + "_lr")
    out = model.run(inter_data)
    return out, data_shape1, data_shape2

def compile_to_pi(target, padded, mask, model, clf, hardware, data_shape1, data_shape2):
    dev = tvm.device(str(target), 0)
    input_ids = torch.tensor(padded, dtype=torch.long)
    attention_mask = torch.tensor(mask)
    scripted_model = torch.jit.trace(model, [input_ids, attention_mask], strict=False)
    shape_list = [("input_ids", input_ids.shape), ("attention_mask", attention_mask.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    lib.export_library(hardware + "_bert.tar")
    
    data = relay.var("data", shape=data_shape1, dtype="float32")
    out = relay.take(data, relay.const(0, "int32"), axis=1)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([data], out)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target)
    lib.export_library(hardware + "_take.tar")

    model = build_model(clf, data_shape2, batch_size=1, target=target, sparse_replacing=False, dtype_converting=False, elimination=True)
    model.save_model(hardware + "_lr")
    return 0

out, data_shape1, data_shape2 = compile_to_lib(target[0], padded, mask, model, clf, hardware="cpu")
compile_to_pi(target[1], padded, mask, model, clf, "pi", data_shape1, data_shape2)
#print(out)
