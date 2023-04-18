import pandas as pd
import numpy as np
import pickle
import torch
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from cmlcompiler.model import GBDTFeature
import sys
from copy import deepcopy

test_data_path = "avazu/test_data.csv"
gbdt_model_path = "avazu/gbdt_model"
wide_deep_model_path = "avazu/wide_deep.pt"

def compile_to_lib():
    """
    Compile pipelines to both server cpu and raspberrypi
    """
    hardware = ["cpu", "pi"]
    target = ["llvm -mcpu=core-avx2", "llvm -model=bcm2711 -mtriple=armv8l-linux-gnueabihf -mattr=+neon -mcpu=cortex-a72"]
    X = np.loadtxt(test_data_path, dtype=int, delimiter=",")
    # save auxiliary array
    gbdt_sklearn = pickle.load(open(gbdt_model_path, 'rb'))
    batch_size = 1 
    encoder_shape = sum([len(i) for i in gbdt_sklearn.encoder.categories_]) 
    encoder_array = np.zeros((batch_size, encoder_shape))
    feature = np.zeros((batch_size, X.shape[0]))
    feature[:] = X
    np.savetxt("encoder_array", encoder_array, fmt="%d")
    #Compile GBDT model
    #Compile pi without running
    gbdt_sklearn_copy = deepcopy(gbdt_sklearn)
    gbdt_model = GBDTFeature(gbdt_sklearn_copy, X.shape[0], target[1], "float32", "int32", "forest_reg", 1, False, False, False, False)
    gbdt_model.build()
    gbdt_model.save_model(hardware[1]+"_gbdt")
    #Compile cpu and run
    gbdt_model = GBDTFeature(gbdt_sklearn, X.shape[0], target[0], "float32", "int32", "forest_reg", 1, False, False, False, False)
    gbdt_model.build()
    gbdt_model.save_model(hardware[0]+"_gbdt")
    X = np.array(X, dtype="float32") 
    encoder_index = gbdt_model.run(tvm.nd.array(X)) 
    encoder_index = encoder_index.asnumpy()
    encoder_array[0][encoder_index[0]] = 1
    X_feature = np.concatenate((feature, encoder_array), axis=-1)
    #Compile WideAndDeep model
    X_feature = torch.tensor(X_feature, dtype=torch.long)
    wide_deep_torch = torch.load(wide_deep_model_path)
    scripted_model = torch.jit.trace(wide_deep_torch, [X_feature], strict=False)
    shape_list = [("X_feature", X_feature.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target[0], params=params)
    lib.export_library(hardware[0]+"_wide_deep.tar")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target[1], params=params)
    lib.export_library(hardware[1]+"_wide_deep.tar")
    return 0

compile_to_lib()