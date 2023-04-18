import tvm
from tvm import relay

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
