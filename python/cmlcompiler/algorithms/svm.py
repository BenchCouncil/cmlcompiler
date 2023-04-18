"""svm models"""
import tvm
from tvm import relay
from cmlcompiler.algorithms.linear import base_classification,base_regression 

def linear_svc(data_shape, n_class, dtype="float32"):
    """
    Linear Support Vector Classifier
    data (batch_size, n_feature)
    weight (n_class, n_feature)
    bias (n_class, )
    """
    return base_classification(data_shape, n_class, dtype)

def linear_svr(data_shape, n_class, dtype="float32"):
    """
    Linear Support Vector Regressor
    data (batch_size, n_feature)
    weight (1, n_feature)
    bias (1, )
    """
    return base_regression(data_shape, n_class, dtype)

def linear_kernel_svr(data_shape, n_sv, dtype="float32"):
    """
    svr with linear as kernel
    kernel function: <x, support_vectors>
    n_sv: number of support vectors
    data (batch_size, n_feature)
    support_vectors (n_sv, n_feature)
    dual_coef (1, n_sv)
    bias (1, )
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    kernel = relay.nn.dense(data, relay.var("support_vectors"), units=n_sv)
    y = relay.nn.dense(kernel, relay.var("dual_coef"), units=1)
    y = relay.nn.bias_add(y, relay.var("bias"), axis=-1)
    return y

def sigmoid_kernel_svr(data_shape, n_sv, dtype="float32"):
    """
    svr with sigmoid as kernel
    kernel function: tanh(<x, support_vectors> + coef0)
    n_sv: number of support vectors
    data (batch_size, n_feature)
    coef0 (1, )
    support_vectors (n_sv, n_feature)
    dual_coef (1, n_sv)
    bias (1, )
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    kernel = relay.nn.dense(data, relay.var("support_vectors"), units=n_sv)
    coef0 = relay.var("coef0", shape=(1,))
    kernel = relay.add(kernel, coef0)
    kernel = relay.tanh(kernel)
    y = relay.nn.dense(kernel, relay.var("dual_coef"), units=1)
    y = relay.nn.bias_add(y, relay.var("bias"), axis=-1)
    return y

def poly_kernel_svr(data_shape, n_sv, dtype="float32"):
    """
    svr with poly as kernel
    kernel function: (<x, support_vectors> + coef0) ^ degree
    n_sv: number of support vectors
    data (batch_size, n_feature)
    coef0 (1, )
    degree (1,)
    support_vectors (n_sv, n_feature)
    dual_coef (1, n_sv)
    bias (1, )
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    kernel = relay.nn.dense(data, relay.var("support_vectors"), units=n_sv)
    coef0 = relay.var("coef0", shape=(1,))
    kernel = relay.add(kernel, coef0)
    degree = relay.var("degree", shape=(1,))
    kernel = relay.power(kernel, degree)
    y = relay.nn.dense(kernel, relay.var("dual_coef"), units=1)
    y = relay.nn.bias_add(y, relay.var("bias"), axis=-1)
    return y

def rbf_kernel_svr(data_shape, n_sv, dtype="float32"):
    """
    svr with rbf as kernel
    kernel function: exp(gamma * x^2 + sv_norm + <x, support_vectors>)
    n_sv: number of support vectors
    data (batch_size, n_feature)
    support_vectors (n_sv, n_feature)
    sv_norm (1, n_sv)
    dual_coef (1, n_sv)
    bias (1, )
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    norm = relay.power(data, relay.const(2.0))
    norm = relay.sum(norm, axis=-1)
    gamma = relay.var("gamma", shape=(1,))
    norm = relay.multiply(norm, gamma)
    norm = relay.reshape(norm, newshape=[-1, 1])
    kernel = relay.nn.dense(data, relay.var("support_vectors"), units=n_sv)
    kernel = relay.add(kernel, norm)
    sv_norm = relay.var("sv_norm", shape=(1, n_sv))
    kernel = relay.add(kernel, sv_norm)
    kernel = relay.exp(kernel)
    y = relay.nn.dense(kernel, relay.var("dual_coef"), units=1)
    y = relay.nn.bias_add(y, relay.var("bias"), axis=-1)
    return y

def svc(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors, dual_coef, bias, n_support):
    """
    Support Vector Classifier
    x [n_samples, n_features]
    support_vectors [n_features, n_sv]
    dual_coef [n_sv, n_classes-1]
    bias [n_classes*(n_classes-1)/2, ]
    Output [n_samples,]
    """
    n_samples = x.shape[0]
    n_sv = support_vectors.shape[1]
    # kernel [n_samples, n_sv]
    kernel = kernel_func(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors)
    n_bias = bias.shape[0]
    n_classes = bias.shape[0]
    if(n_classes == 1):
        y = topi.matmul(kernel, dual_coef)
        y = te.compute((n_samples, n_classes), lambda i, j: y[i][j] + bias[j])
        y = topi.greater(y, 0).astype("int8")
    else:
        #TODO: Add support for multi-class classification
        #y = te.compute((n_samples, n_bias), lambda m, n: 
        #        te.compute(kernel[m][k1] * dual_coef[k1][j-1] + kernel[m][k2] * dual_coef[k2][i] + bias[n]))
        pass
    return y

def linear_kernel_svc(data_shape, n_sv, dtype="float32"):
    """
    svc with linear as kernel
    kernel function: <x, support_vectors>
    n_sv: number of support vectors
    data (batch_size, n_feature)
    support_vectors (n_sv, n_feature)
    dual_coef (1, n_sv)
    bias (1, )
    """
    data = relay.var("data", shape=data_shape, dtype=dtype)
    kernel = relay.nn.dense(data, relay.var("support_vectors"), units=n_sv)
    y = relay.nn.dense(kernel, relay.var("dual_coef"), units=1)
    y = relay.nn.bias_add(y, relay.var("bias"), axis=-1)
    return y

def sigmoid_kernel_svc(data_shape, n_sv, dtype="float32"):
    return 0

def poly_kernel_svc(data_shape, n_sv, dtype="float32"):
    return 0

def rbf_kernel_svc(data_shape, n_sv, dtype="float32"):
    return 0

