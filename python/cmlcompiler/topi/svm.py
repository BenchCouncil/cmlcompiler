"""svm models"""
import tvm
from tvm import te, topi, tir

def linear_svc(x, coef, bias):
    """
    Linear Support Vector Classifier
    x [n_samples, n_features]
    coef [n_features, n_classes]
    bias [n_classes, ]
    Output [n_samples,]
    """
    n_samples = x.shape[0]
    n_features = x.shape[1]
    n_classes = bias.shape[0]
    y = topi.matmul(x, coef)
    y = te.compute((n_samples, n_classes), lambda i, j: y[i][j] + bias[j])
    if(n_classes == 1):
        y = topi.greater(y, 0).astype("int8")
    else:
        y = topi.argmax(y, axis=1)
    return y

def linear_svr(x, coef, bias):
    """
    Linear Support Vector Regressor
    x [n_samples, n_features]
    coef [n_features, n_targets]
    bias [n_targets, ]
    Output [n_samples, n_targets]
    """
    n_samples = x.shape[0]
    n_features = x.shape[1]
    if(len(coef.shape) == 1):
        k = te.reduce_axis((0, n_features))
        y = te.compute((n_samples, 1), lambda i: te.sum(x[i][k] * coef[k], axis = k))
        y = topi.add(y, bias)
    else:
        y = topi.matmul(x, coef)
        n_targets = coef.shape
        y = te.compute((n_samples, n_targets), lambda i, j: y[i][j] + bias[j])
    return y

def kernel_func(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors):
    """
    Using the kernel function to calculate the kernel
    """
    if kernel_shape == "linear":
        # kernel function: <x, x'>
        kernel = topi.matmul(x, support_vectors)
    elif kernel_shape == "sigmoid": 
        # kernel function: tanh(gamma * <x, x'> + coef0)
        kernel = topi.matmul(x, support_vectors)
        kernel = te.compute(kernel.shape, lambda i, j:
                te.tanh(gamma * kernel[i][j] + coef0))
    elif kernel_shape == "poly":
        # kernel function: (gamma * <x, x'>) ^ degree
        kernel = topi.matmul(x, support_vectors)
        kernel = te.compute(kernel.shape, lambda i, j:
                te.power(gamma * kernel[i][j] + coef0, degree))
    elif kernel_shape == "rbf":
        # kernel function: exp(-gamma * ||x - x'||2)
        kernel = topi.matmul(x, support_vectors)
        # x_norm [n_samples,]
        n_samples = x.shape[0]
        n_features = x.shape[1]
        k = te.reduce_axis((0, n_features))
        x_norm = te.compute((n_samples,), lambda i: te.sum(te.power(x[i][k], 2), axis = k))   
        kernel = te.compute(kernel.shape, lambda i, j:
                te.exp(-gamma * (x_norm[i] - 2 * kernel[i][j] + sv_norm[j])))
    return kernel

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

def svr(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors, dual_coef, bias, n_support):
    """
    Support Vector Regressor
    x [n_samples, n_features]
    support_vectors [n_features, n_sv]
    dual_coef [n_sv, 1]
    bias [1, ]
    Output [n_samples,]
    """
    n_samples = x.shape[0]
    n_sv = support_vectors.shape[1]
    # kernel [n_samples, n_sv]
    kernel = kernel_func(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors)
    if(len(bias.shape) == 1):
        k = te.reduce_axis((0, n_sv))
        y = te.compute((n_samples, 1), lambda i: te.sum(kernel[i][k] * dual_coef[k][0], axis = k))
        y = topi.add(y, bias)
    else:
        y = topi.matmul(kernel, dual_coef)
        n_targets = dual_coef.shape
        y = te.compute((n_samples, n_targets), lambda i, j: y[i][j] + bias[j])
    y = kernel
    return y

def nu_svc(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors, dual_coef, bias, n_support):
    """
    Nu-Support Vector Classifier, using a parameter to control the number of support vectors
    """
    return svc(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors, dual_coef, bias, n_support)

def nu_svr(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors, dual_coef, bias, n_support):
    """
    Nu-Support Vector Regressor, using a parameter to control the number of support vectors
    """
    return svr(x, kernel_shape, gamma, coef0, degree, sv_norm, support_vectors, dual_coef, bias, n_support)

