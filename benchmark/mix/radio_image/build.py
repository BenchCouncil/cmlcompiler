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

simpleDNN_path = "simpleDNN.pt"
forest_path = "forest"
simple_feature_path = "simple_feature.pt"

def convert_torch_model(model):
    pretrained_dict = model.state_dict()
    new_model = simple_feature()
    model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    new_model.load_state_dict(model_dict)
    return new_model

def compile_to_lib():
    hardware = ["cpu", "pi"]
    target = ["llvm -mcpu=core-avx2", "llvm -model=bcm2711 -mtriple=armv8l-linux-gnueabihf -mattr=+neon -mcpu=cortex-a72"]
    simple_feature = torch.load(simple_feature_path)
    X = torch.load("single_sample.pt")
    scripted_model = torch.jit.trace(simple_feature, [X], strict=False)
    shape_list = [("X", X.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    # Compile to pi without running
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target[1], params=params)
    lib.export_library(hardware[1] + "_DNN.tar")
    #Compile cpu and run
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target[0], params=params)
    lib.export_library(hardware[0] + "_DNN.tar")
    dev = tvm.device(str(target[0]), 0)    
    feature_model = graph_executor.GraphModule(lib["default"](dev))
    feature_model.set_input("X", X)
    feature_model.run()
    X_feature = feature_model.get_output(0)
    forest = pickle.load(open(forest_path, 'rb'))
    forest_model = build_model(forest, X_feature.shape, batch_size=1, target=target[1], sparse_replacing=False, dtype_converting=False, elimination=True)
    forest_model.save_model(hardware[1]+"_forest")
    
    forest_model = build_model(forest, X_feature.shape, batch_size=1, target=target[0], sparse_replacing=False, dtype_converting=False, elimination=True)
    forest_model.save_model(hardware[0]+"_forest")
    #out = forest_model.run(X_feature)
    return 0

#simple_dnn = torch.load(simpleDNN_path)
#simple_feature = convert_torch_model(simple_dnn)
#torch.save(simple_feature, simple_feature_path)
compile_to_lib()    
