"""linear models"""
import tvm
from tvm import te, topi, tir
from cmlcompiler.topi.x86.dense import dense_pack

def base_classification(x, coef, bias, dtype):
    """
    Base Classifier
    x [n_samples, n_features]
    coef [n_features, n_classes]
    bias [n_classes, ]
    Output [n_samples,]
    """
    y = dense_pack(x, coef, bias, dtype)
    #y = topi.x86.dense_pack(x, coef, bias, dtype)  
    return topi.argmax(y, axis=1)

def logistic_regression(x, coef, bias, dtype):
    """
    Logistic Regression Classifier
    Note that it's not for regression, but for classification
    """
    return base_classification(x, coef, bias, dtype)

def logistic_regression_cv(x, coef, bias, dtype):
    """
    Logistic Regression Classifier with built-in cross-validation
    Note that it's not for regression, but for classification
    """
    return base_classification(x, coef, bias, dtype)

def ridge_classifier(x, coef, bias, dtype):
    """
    Ridge classifier
    """
    return base_classification(x, coef, bias, dtype)

def ridge_classifier_cv(x, coef, bias, dtype):
    """
    Ridge classifier with built-in cross-validation
    """
    return base_classification(x, coef, bias, dtype)

def sgd_classifier(x, coef, bias, dtype):
    """
    SGD classifier
    """
    return base_classification(x, coef, bias, dtype)

def perceptron(x, coef, bias, dtype):
    """
    perceptron
    """
    return base_classification(x, coef, bias, dtype)

@tvm.te.tag_scope(tag="regression_single_bias_output")
def regression_single_bias(x, coef, bias, dtype):
    """
    bias is a single variable
    """
    n_samples = x.shape[0]
    n_features = x.shape[1]
    k = te.reduce_axis((0, n_features))
    y = te.compute((n_samples, 1), lambda i: te.sum(x[i][k] * coef[k], axis = k))
    y = topi.add(y, bias)
    return y

@tvm.te.tag_scope(tag="regression_vector_bias_output")
def regression_vector_bias(x, coef, bias, dtype):
    """
    bias is a vector
    """
    n_samples = x.shape[0]
    n_features = x.shape[1]
    y = topi.matmul(x, coef)
    n_targets = coef.shape
    y = te.compute((n_samples, n_targets), lambda i, j: y[i][j] + bias[j])
    return y

def base_regression(x, coef, bias, dtype):
    """
    Base Regressor
    x [n_samples, n_features]
    coef [n_features, n_targets]
    bias [n_targets, ]
    Output [n_samples, n_targets]
    """
    if(len(coef.shape) == 1):
        y = regression_single_bias(x, coef, bias, dtype)
    else:
        y = regression_vector_bias(x, coef, bias, dtype)
    return y


def linear_regression(x, coef, bias, dtype):
    """
    Linear Regression
    """
    return base_regression(x, coef, bias, dtype)

def ridge(x, coef, bias, dtype):
    """
    Ridge Regressor
    """
    return base_regression(x, coef, bias, dtype)

def ridge_cv(x, coef, bias, dtype):
    """
    Ridge Regressor with built-in cross-validation
    """
    return base_regression(x, coef, bias, dtype)

def sgd_regressor(x, coef, bias, dtype):
    """
    SGD Regressor
    """
    return base_regression(x, coef, bias, dtype)

