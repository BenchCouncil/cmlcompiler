import tvm
from tvm import relay

def binarizer(data_shape, dtype="float32"):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    threshold = relay.var("threshold", shape=(1,), dtype=dtype)
    y = relay.greater(data, threshold)
    return y

def normalizer(data_shape, dtype="float32", norm="l2"):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    if(norm == "l1"):
        y = relay.abs(data)
        y = relay.sum(y, axis=-1, keepdims=True)
        y = relay.divide(data, y)
    elif(norm == "max"):
        y = relay.abs(data)
        y = relay.max(data, axis=-1, keepdims=True)
        y = relay.divide(data, y)
    elif(norm == "l2"):
        #y = relay.nn.l2_normalize(data, eps=0, axis=[1])
        y = relay.power(data, relay.const(2.0))
        y = relay.sum(y, axis=-1, keepdims=True)
        y = relay.sqrt(y)
        y = relay.divide(data, y)
    return y

def label_encoder(data_shape, dtype="float32"):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    unique, indices, inverse_indices, num_unique = relay.unique(data, is_sorted=True, return_counts=False)
    return inverse_indices
