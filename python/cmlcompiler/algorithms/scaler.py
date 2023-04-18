import tvm
from tvm import relay

def min_max_scaler(data_shape, n_feature, dtype="float32"):
    """
    Min Max Scaler
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    scale = relay.var("scale", shape=(1, n_feature), dtype=dtype)
    min_x = relay.var("min", shape=(n_feature,), dtype=dtype)
    y = relay.multiply(data, scale)
    y = relay.nn.bias_add(y, min_x)
    return y
 
def max_abs_scaler(data_shape, n_feature, dtype="float32"):
    """
    Max Abs Scaler
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    scale = relay.var("scale", shape=(n_feature,), dtype=dtype)
    y = relay.divide(data, scale)
    return y
 
def standard_scaler(data_shape, n_feature, dtype="float32"):
    """
    Standard Scaler
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    mean = relay.var("mean", shape=(n_feature,), dtype=dtype)
    scale = relay.var("scale", shape=(n_feature,), dtype=dtype)
    y = relay.subtract(data, mean)
    y = relay.divide(y, scale)
    return y

def robust_scaler(data_shape, n_feature, dtype="float32"):
    """
    Robust Scaler
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    center = relay.var("center", shape=(n_feature,), dtype=dtype)
    scale = relay.var("scale", shape=(n_feature,), dtype=dtype)
    y = relay.subtract(data, center)
    y = relay.divide(y, scale)
    return y
