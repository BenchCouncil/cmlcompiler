import time
from hummingbird.ml import convert
from sklearn.datasets import make_classification,make_multilabel_classification,make_regression
import tvm
import tvm.testing
import numpy as np

def bench_sklearn(model, input_data, number, check_flag=False):
    """
    Benchmarking sklearn model
    Input model and data
    Return average exection time 
    """
    input_data = np.array(input_data)
    #if(model.predict):
    #    model.transform = model.predict
    try:
        model.transform = model.predict
    except Exception as e:
        pass
    start_time = time.perf_counter()
    for i in range(number):
        out_data = model.transform(input_data)
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / number
    if(check_flag == True):
        return avg_time, out_data
    else:
        return avg_time

def bench_hb(model, input_data, number, check_flag, backend="pytorch", target="llvm"):
    """
    Benchmarking hummingbird model
    Input model and data
    Return average exection time 
    """
    if(target == "cuda"):
        if(backend == "pytorch"):
            hb_model = convert(model, backend).to("cuda")
        elif(backend == "tvm"):
            hb_model = convert(model, backend, input_data, "cuda")
    else:
        if(backend == "pytorch"):
            hb_model = convert(model, backend)
        elif(backend == "tvm"):
            hb_model = convert(model, backend, input_data)
    if(hb_model.predict):
        hb_model.transform = hb_model.predict
    start_time = time.perf_counter()
    for i in range(number):
        out_data = hb_model.transform(input_data)
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / number
    if(check_flag == True):
        return avg_time, out_data
    else:
        return avg_time

def bench_hb_all(model, input_data, number, check_flag):
    """
    Benchmarking hummingbird model in four types: pytorch-cpu, pytorch-gpu, tvm-cpu, tvm-gpu
    """
    input_data = np.array(input_data)
    if(check_flag == True):
        hb_py, out_hb_py = bench_hb(model, input_data, number=number, check_flag=check_flag, backend="pytorch")
        hb_py_gpu, out_hb_py_gpu = bench_hb(model, input_data, number=number, check_flag=check_flag, backend="pytorch", target="cuda")
        hb_tvm, out_hb_tvm = bench_hb(model, input_data, number=number, check_flag=check_flag, backend="tvm")
        try:
            hb_tvm_gpu, out_hb_tvm_gpu = bench_hb(model, input_data, number=number, check_flag=check_flag, backend="tvm", target="cuda")
        except Exception as e:
            print(e)
            hb_tvm_gpu = -1
            pass
        return hb_py, hb_py_gpu, hb_tvm, hb_tvm_gpu, out_hb_py, out_hb_py_gpu, out_hb_tvm, out_hb_tvm_gpu
    else:
        try:
            hb_py = bench_hb(model, input_data, number=number, check_flag=check_flag, backend="pytorch")
        except Exception as e:
            print(e)
            hb_py = -1
            pass
        try:
            hb_py_gpu = bench_hb(model, input_data, number=number, check_flag=check_flag, backend="pytorch", target="cuda")
        except Exception as e:
            print(e)
            hb_py_gpu = -1
            pass
        print("hummingbird pytorch")
        try:
            hb_tvm = bench_hb(model, input_data, number=number, check_flag=check_flag, backend="tvm")
        except Exception as e:
            print(e)
            hb_tvm = -1
            pass
        print("hummingbird tvm cpu")
        try:
            hb_tvm_gpu = bench_hb(model, input_data, number=number, check_flag=check_flag, backend="tvm", target="cuda")
        except Exception as e:
            print(e)
            hb_tvm_gpu = -1
            pass
        print("hummingbird tvm gpu")
        return hb_py, hb_py_gpu, hb_tvm, hb_tvm_gpu

def bench_cmlcompiler(model, number, input_data, check_flag=False):
    """
    Benchmarking cmlcompiler model
    Input model and data
    Return average exection time 
    """

    input_data = tvm.nd.array(input_data)
    start_time = time.perf_counter()
    for i in range(number):
        try:
            load_time, exec_time, store_time, take_time, out_data = model.run(input_data, breakdown=True)
        except Exception as e:
            print(e)
            load_time = exec_time = store_time = take_time = 0
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / number
    if(check_flag == True):
        return load_time, exec_time, store_time, take_time, avg_time, out_data
    else:
        return load_time, exec_time, store_time, take_time, avg_time

def generate_dataset(n_samples, n_features, n_classes, n_labels=1, regression=True):
    if(regression == True):
        X, y = make_regression(n_samples=n_samples, n_features=n_features)
    else:
        if(n_labels == 1):
            n_informative = int(n_features / 2)
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative)
        else:
            X, y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_labels=n_labels)
    return X, y 

def check_data(ground_truth, y, rtol=1e-4):
    """
    While benchmarking, always using sklearn results as ground truth
    y: data to be checked
    """
    try:
        tvm.testing.assert_allclose(ground_truth, y, rtol=1e-4)
        print("pass")
        return True
    except Exception as e:
        print("error")
        print(e)
        return False
