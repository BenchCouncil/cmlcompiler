"""ensemble models based on gemm"""
import tvm
from tvm import relay
from collections import namedtuple

def random_forest_classifier(
        data_shape, 
        n_estimator_x_internal_node,
        n_estimator,
        batch_size,
        internal_node,
        leaf_node,
        n_estimator_x_leaf_node, 
        label, 
        dtype_converting,
        sparse_replacing,
        dtype="float32"
        ):
    """
    forest implement based on dense gemm
    data [batch_size, n_feature]  //dense float matrix
    S [n_estimator * internal_node, n_feature] //sparse 0-1 matrix, sparisty = 1 / n_feature
    T [n_estimator * internal_node, 1] //dense float vector
    B [n_estimator, leaf_node, internal_node]
    //sparse int matrix, sparisty differs with tree structure, range to be limited 
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    # TODO: reduce reshape, when batch_size is larger, the cost of reshape is horrible
    if(dtype_converting == True):
        min_dtype = "int8"
        #min_dtype = "int16"
    else:
        min_dtype = dtype
    index_dtype = dtype
    # How about converting batch matmul to matmul?
    # [batch_size, n_feature]
    if(sparse_replacing == True):
        S_data = relay.var("S_data", dtype=dtype)
        S_indices = relay.var("S_indices", dtype=index_dtype)
        S_indptr = relay.var("S_indptr", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "indices", "indptr"])
        S = Sparse(S_data, S_indices, S_indptr)
        y = relay.nn.sparse_dense(data, S)
    else:
        y = relay.nn.dense(data, relay.var("S", dtype=dtype), units=n_estimator_x_internal_node)
    # [batch_size, n_estimator * internal_node]
    y = relay.greater(y, relay.var("T", shape=(n_estimator_x_internal_node,)))
    # [batch_size, n_estimator * internal_node]
    y = relay.cast(y, min_dtype)
    y = relay.reshape(y, (batch_size, n_estimator, internal_node))
    # [batch_size, n_estimator, internal_node]
    y = relay.transpose(y, axes=[1, 0, 2])
    # Noting that reshape directly cause error
    # [n_estimator, batch_size, internal_node]
    b = relay.var("B", shape=(n_estimator, leaf_node, internal_node))
    # should be batch matmul, rather than matmul
    #y = relay.nn.batch_matmul(y, c, out_dtype=min_dtype)
    y = relay.nn.batch_matmul(y, b)
    # [n_estimator, batch_size, leaf_node]
    y = relay.argmax(y, axis=-1)
    # [n_estimator, batch_size]
    y = relay.transpose(y, axes=[1, 0])
    # [batch_size, n_estimator]
    # Using step to index n estimator
    y = relay.cast(y, "int32")
    step = relay.var("step", shape=(n_estimator,))
    y = relay.add(y, step)
    l = relay.var("L", shape = (n_estimator_x_leaf_node, label))
    #y = relay.cast(y, "int32")
    y = relay.take(l, y, axis=0)
    y = relay.sum(y, axis=1)
    y = relay.argmax(y, axis=-1)
    classes = relay.var("classes", shape=(label,))
    y = relay.take(classes, y)
    return y 

def extra_trees_classifier(
        data_shape, 
        n_estimator_x_internal_node, 
        n_estimator, 
        batch_size,
        internal_node, 
        leaf_node,
        n_estimator_x_leaf_node, 
        label, 
        dtype_converting,
        sparse_replacing,
        dtype="float32"
        ):
    """
    Decision tree classifier
    """
    y = random_forest_classifier(
            data_shape, 
            n_estimator_x_internal_node, 
            n_estimator,
            batch_size,
            internal_node, 
            leaf_node,
            n_estimator_x_leaf_node, 
            label, 
            dtype_converting,
            sparse_replacing,
            dtype="float32"
            )
    #y = relay.argmax(y, axis=1)
    # [batch_size, 1]
    return y

def random_forest_regressor(
        data_shape, 
        n_estimator_x_internal_node,
        n_estimator,
        batch_size,
        internal_node,
        leaf_node,
        n_estimator_x_leaf_node, 
        label, 
        dtype_converting,
        sparse_replacing,
        dtype="float32"
        ):
    """
    forest implement based on dense gemm
    data [batch_size, n_feature]  //dense float matrix
    S [n_estimator * internal_node, n_feature] //sparse 0-1 matrix, sparisty = 1 / n_feature
    T [n_estimator * internal_node, 1] //dense float vector
    B [n_estimator, leaf_node, internal_node]
    //sparse int matrix, sparisty differs with tree structure, range to be limited 
    """
    # TODO: reduce reshape, when batch_size is larger, the cost of reshape is horrible
    data = relay.var("data", shape=data_shape, dtype=dtype)
    if(dtype_converting == True):
        min_dtype = "int8"
        #min_dtype = "int16"
    else:
        min_dtype = dtype
    index_dtype = dtype
    # How about converting batch matmul to matmul?
    # [batch_size, n_feature]
    if(sparse_replacing == True):
        S_data = relay.var("S_data", dtype=dtype)
        S_indices = relay.var("S_indices", dtype=index_dtype)
        S_indptr = relay.var("S_indptr", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "indices", "indptr"])
        S = Sparse(S_data, S_indices, S_indptr)
        y = relay.nn.sparse_dense(data, S)
    else:
        y = relay.nn.dense(data, relay.var("S", dtype=dtype), units=n_estimator_x_internal_node)
    # [batch_size, n_estimator * internal_node]
    y = relay.greater(y, relay.var("T", shape=(n_estimator_x_internal_node,)))
    # [batch_size, n_estimator * internal_node]
    y = relay.cast(y, min_dtype)
    y = relay.reshape(y, (batch_size, n_estimator, internal_node))
    # [batch_size, n_estimator, internal_node]
    y = relay.transpose(y, axes=[1, 0, 2])
    # Noting that reshape directly cause error
    # [n_estimator, batch_size, internal_node]
    b = relay.var("B", shape=(n_estimator, leaf_node, internal_node))
    # should be batch matmul, rather than matmul
    #y = relay.nn.batch_matmul(y, c, out_dtype=min_dtype)
    y = relay.nn.batch_matmul(y, b)
    # [n_estimator, batch_size, leaf_node]
    y = relay.argmax(y, axis=-1)
    # [n_estimator, batch_size]
    y = relay.transpose(y, axes=[1, 0])
    # [batch_size, n_estimator]
    # Using step to index n estimator
    y = relay.cast(y, "int32")
    step = relay.var("step", shape=(n_estimator,))
    y = relay.add(y, step)
    l = relay.var("L", shape = (n_estimator_x_leaf_node, label))
    #y = relay.cast(y, "int32")
    y = relay.take(l, y, axis=0)
    y = relay.mean(y, axis=1)
    return y 

def extra_trees_regressor(
        data_shape, 
        n_estimator_x_internal_node, 
        n_estimator, 
        batch_size,
        internal_node, 
        leaf_node,
        n_estimator_x_leaf_node, 
        label, 
        dtype_converting,
        sparse_replacing,
        dtype="float32"
        ):
    """
    Decision tree classifier
    """
    y = random_forest_regressor(
            data_shape, 
            n_estimator_x_internal_node, 
            n_estimator,
            batch_size,
            internal_node, 
            leaf_node,
            n_estimator_x_leaf_node, 
            label, 
            dtype_converting,
            sparse_replacing,
            dtype="float32"
            )
    #y = relay.argmax(y, axis=1)
    # [batch_size, 1]
    return y

def forest_feature_gemm_dense(
        data_shape, 
        n_estimator_x_internal_node,
        n_estimator,
        batch_size,
        internal_node,
        leaf_node,
        n_estimator_x_leaf_node, 
        label, 
        dtype_converting,
        sparse_replacing,
        dtype="float32"
        ):
    """
    forest implement based on dense gemm
    data [batch_size, n_feature]  //dense float matrix
    S [n_estimator * internal_node, n_feature] //sparse 0-1 matrix, sparisty = 1 / n_feature
    T [n_estimator * internal_node, 1] //dense float vector
    B [n_estimator, leaf_node, internal_node]
    //sparse int matrix, sparisty differs with tree structure, range to be limited 
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    # TODO: reduce reshape, when batch_size is larger, the cost of reshape is horrible
    if(dtype_converting == True):
        min_dtype = "int8"
        #min_dtype = "int16"
    else:
        min_dtype = dtype
    index_dtype = dtype
    # How about converting batch matmul to matmul?
    # [batch_size, n_feature]
    if(sparse_replacing == True):
        S_data = relay.var("S_data", dtype=dtype)
        S_indices = relay.var("S_indices", dtype=index_dtype)
        S_indptr = relay.var("S_indptr", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "indices", "indptr"])
        S = Sparse(S_data, S_indices, S_indptr)
        y = relay.nn.sparse_dense(data, S)
    else:
        y = relay.nn.dense(data, relay.var("S", dtype=dtype), units=n_estimator_x_internal_node)
    # [batch_size, n_estimator * internal_node]
    y = relay.greater(y, relay.var("T", shape=(n_estimator_x_internal_node,)))
    # [batch_size, n_estimator * internal_node]
    y = relay.cast(y, min_dtype)
    y = relay.reshape(y, (batch_size, n_estimator, internal_node))
    # [batch_size, n_estimator, internal_node]
    y = relay.transpose(y, axes=[1, 0, 2])
    # Noting that reshape directly cause error
    # [n_estimator, batch_size, internal_node]
    b = relay.var("B", shape=(n_estimator, leaf_node, internal_node))
    # should be batch matmul, rather than matmul
    #y = relay.nn.batch_matmul(y, c, out_dtype=min_dtype)
    y = relay.nn.batch_matmul(y, b)
    # [n_estimator, batch_size, leaf_node]
    y = relay.argmax(y, axis=-1)
    # [n_estimator, batch_size]
    y = relay.transpose(y, axes=[1, 0])
    cumsum = relay.var("cumsum", shape=(n_estimator, 1))
    y = relay.add(y, cumsum)
    #zero_array = relay.zeros((n_estimator, ),dtype="int32")
    #out = relay.scatter_add(zero_array, y, relay.Constant(tvm.nd.array(1)), 0)
    return y



