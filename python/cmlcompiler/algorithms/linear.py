"""linear models, based on relay"""
import tvm
from tvm import relay

def linear_binary_classification(data_shape, n_class, dtype="float32"):
    """
    linear binary classification
    data (batch_size, n_feature)
    weight (1, n_feature)
    bias (1, )
    Using one var to represent 0-1 class, differs from multi classification
    """
    #TODO: Add thin-and-tall matmul to topi and register in relay
    data = relay.var("data", shape=data_shape, dtype="float32")
    y = relay.nn.dense(data, relay.var("weight"), units=n_class)
    y = relay.nn.bias_add(y, relay.var("bias"), axis=-1)
    y = relay.greater(y, relay.const(0.0))
    return y

def base_classification(data_shape, n_class, elimination, dtype="float32"):
    """
    linear based classification
    data (batch_size, n_feature)
    weight (n_class, n_feature)
    bias (n_class, )
    """
    #TODO: Add thin-and-tall matmul to topi and register in relay
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=(n_class, data_shape[1]), dtype=dtype)
    y = relay.nn.dense(data, relay.var("weight"), units=n_class)
    bias = relay.var("bias", shape=(n_class,), dtype=dtype)
    y = relay.nn.bias_add(y, bias, axis=-1)
    if(elimination == False):
        y = relay.nn.softmax(y, axis=1)
    y = relay.argmax(y, axis=1)
    #classes = relay.var("classes", shape=(n_class,), dtype=dtype)
    y = relay.cast(y, "int32")
    classes = relay.var("classes")
    # Note that y is set as rhs in relay.take, influence the order of params in relay.build
    y = relay.take(classes, y)
    return y

def logistic_regression(data_shape, n_class, elimination, dtype="float32"):
    """
    Logistic Regression Classifier
    Note that it's not for regression, but for classification
    """
    return base_classification(data_shape, n_class, elimination, dtype)

def logistic_regression_cv(data_shape, n_class, elimination, dtype="float32"):
    """
    Logistic Regression Classifier with built-in cross-validation
    Note that it's not for regression, but for classification
    """
    return base_classification(data_shape, n_class, elimination, dtype)

def ridge_classifier(data_shape, n_class, elimination, dtype="float32"):
    """
    Ridge classifier
    """
    return base_classification(data_shape, n_class, elimination, dtype)

def ridge_classifier_cv(data_shape, n_class, elimination, dtype="float32"):
    """
    Ridge classifier with built-in cross-validation
    """
    return base_classification(data_shape, n_class, elimination, dtype)

def sgd_classifier(data_shape, n_class, elimination, dtype="float32"):
    """
    SGD classifier
    """
    return base_classification(data_shape, n_class, elimination, dtype)

def perceptron(data_shape, n_class, elimination, dtype="float32"):
    """
    perceptron
    """
    return base_classification(data_shape, n_class, elimination, dtype)

def base_regression(data_shape, n_class, elimination, dtype="float32"):
    """
    linear based regression
    data (batch_size, n_feature)
    weight (1, n_feature)
    bias (1, )
    """
    #TODO: Add matrix-vector multiplication to topi and relay, replace nn.dense
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.dense(data, relay.var("weight"), units=n_class)
    y = relay.nn.bias_add(y, relay.var("bias"), axis=-1)
    return y

def linear_regression(data_shape, n_class, elimination, dtype="float32"):
    """
    Linear Regression
    """
    return base_regression(data_shape, n_class, elimination, dtype)

def ridge(data_shape, n_class, elimination, dtype="float32"):
    """
    Ridge Regressor
    """
    return base_regression(data_shape, n_class, elimination, dtype)

def ridge_cv(data_shape, n_class, elimination, dtype="float32"):
    """
    Ridge Regressor with built-in cross-validation
    """
    return base_regression(data_shape, n_class, elimination, dtype)

def sgd_regressor(data_shape, n_class, elimination, dtype="float32"):
    """
    SGD Regressor
    """
    return base_regression(data_shape, n_class, elimination, dtype)

