"""testing models"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
import numpy as np
from cmlcompiler.utils.supported_ops import clf_ops,reg_ops,linear_ops
from benchmark.utils import bench_sklearn,bench_hb,bench_hb_all,bench_cmlcompiler,generate_dataset
import time
import os 
import pickle
from cmlcompiler.model import build_model
import tvm.testing

# Add nvcc path
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

def bench_linear_model(sklearn_func, data, model_dir, number, dtype_list, target_list):
    # Load model
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    filename = func_name + ".sav"
    filename = os.path.join(model_dir, filename)
    clf = pickle.load(open(filename, 'rb'))
    data = data.astype("float32")
    #func, data_tvm = convert_from_sklearn(clf, data, dtype=dtype, target=target)
    out = clf.predict(data)
    for target in target_list:
        model = build_model(clf, data.shape, out_dtype="int", target=target)
        for i in range(number):
            out_tvm = model.run(data)
    if(out_tvm.ndim == 2 and out_tvm.shape[1] == 1):
        out_tvm = out_tvm.flatten()
    try:
        tvm.testing.assert_allclose(out_tvm, out, rtol=1e-4)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

def test_models(n_samples, number, model_dir, model_list, dtype_list, target_list):
    # Check data type
    for dtype in dtype_list:
        if dtype not in ["int8", "float16", "float32"]:
            raise Exception("Unsupported data type: " + dtype)
    # Check target
    for target in target_list:
        if target not in ["cuda", "llvm", "llvm -mcpu=core-avx2", "llvm -mpcu=avx512"]:
            raise Exception("Unsupported target: " + target)
    # Check model
    flag_clf = False
    flag_reg = False
    for model in model_list:
        if model in clf_ops:
            flag_clf = True
        elif model in reg_ops:
            flag_reg = True
        else:
            raise Exception("Unknown model " + str(model))
    # Generate data
    print("Generating data\n")
    if flag_clf == True:
        X_clf, y_clf = generate_dataset(n_samples, n_features=100, n_classes=10, n_labels=1, regression=False)
    if flag_reg == True:
        X_reg, y_reg = generate_dataset(n_samples, n_features=100, n_classes=10, n_labels=1, regression=True)
    # Run models
    for model in model_list:
        func_name = str(model).split("\'")[1].split(".")[-1]
        print("testing " + func_name + "\n")
        if model in reg_ops:
            bench_linear_model(model, X_reg, model_dir, number=number, dtype_list=dtype_list, target_list=target_list)
        else:
            bench_linear_model(model, X_clf, model_dir, number=number, dtype_list=dtype_list, target_list=target_list)

test_models(1000, 1, "../../benchmark/operators/linear_models", [LogisticRegression], ["float32"], ["llvm -mcpu=core-avx2", "cuda"])
