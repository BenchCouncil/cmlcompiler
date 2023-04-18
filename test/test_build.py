import numpy as np
import tvm
from tvm import te
from tvm.contrib import graph_runtime
from tvm import relay
from cmlcompiler.algorithms.linear import logistic_regression,linear_regression

def build_model(algo, target, dev, dtype="float32", *param_list):
    """
    Build model

    Parameters

    algo : tvm.relay.Expr
        Algorithm based on relay
    param_list : [Ndarray]
        list of parameters for model, parsing from sklearn
    Returns
    
    model : 
        Executable model to run 
    """
    args = relay.analysis.free_vars(algo)
    net = relay.Function(args, algo)
    mod = tvm.IRModule.from_expr(net)
    mod = relay.transform.InferType()(mod)
    name_list = [v.name_hint for v in mod["main"].params]
    params = {}
    i = 0
    for n in name_list:
        if n == "data":
            pass
        else:
            params[n] = param_list[i]
            i = i + 1
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    model = graph_runtime.GraphModule(lib["default"](dev))
    return model

def run_model(model, data, out_shape, dtype="float32"):
    model.set_input("data", data)
    model.run()
    out = model.get_output(0, tvm.nd.empty(out_shape, dtype))
    return out

def test_linear(func, data_shape, n_class, dtype="float32", out_dtype="float32"):
    data = np.random.uniform(size=data_shape).astype(dtype)
    c = tvm.nd.array(np.random.uniform(size=(n_class, data_shape[-1])).astype(dtype))
    b = tvm.nd.array(np.random.uniform(size=(n_class,)).astype(dtype))
    dev = tvm.cpu(0)
    target = "llvm -mcpu=core-avx2"
    algo = func(data_shape, n_class)
    model = build_model(algo, target, dev, dtype, c, b)
    if func == logistic_regression:
        out = run_model(model, data, (data_shape[0],), out_dtype)
    elif func == linear_regression:
        out = run_model(model, data, (data_shape[0], n_class), out_dtype)
    print(out)

if __name__ == "__main__":
    test_linear(logistic_regression, data_shape=(100, 100), n_class=10, dtype="float32", out_dtype="int")
    test_linear(linear_regression, data_shape=(100, 100), n_class=10, dtype="float32", out_dtype="float32")
