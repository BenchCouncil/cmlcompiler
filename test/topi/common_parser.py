import tvm
from tvm import te,topi,relay
from tvm.topi.utils import get_const_tuple
from tvm.contrib import graph_runtime
import numpy as np
from cmlcompiler.topi.linear import logistic_regression,logistic_regression_cv,ridge_classifier,ridge_classifier_cv,sgd_classifier,perceptron,linear_regression,ridge,ridge_cv,sgd_regressor
#from cmlcompiler.topi.x86.linear import schedule_linear
from cmlcompiler.topi.x86.dense import schedule_classification
from cmlcompiler.topi.x86.linear import base_classification_nopack, schedule_classification_nopack
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor

def parse_linear(clf, dtype="float32", target="llvm"):
    coef = clf.coef_
    bias = clf.intercept_
    coef = coef.astype(dtype)
    bias = bias.astype(dtype)
    #coef = coef.T
    ctx = tvm.context(target, 0)
    coef = tvm.nd.array(coef)
    bias = tvm.nd.array(bias)
    return coef, bias

def parse_svm(clf, dtype="float32", target="llvm"):
    support_vectors = clf.support_vectors_
    dual_coef = clf.dual_coef_
    bias = clf.intercept_
    n_support = clf.n_support_
    support_vectors = support_vectors.astype(dtype)
    dual_coef = dual_coef.astype(dtype)
    bias = bias.astype(dtype)
    n_support = n_support.astype(dtype)
    support_vectors = support_vectors.T
    dual_coef = dual_coef.T
    
    sv_norm = np.power(support_vectors, 2)
    sv_norm = np.sum(sv_norm, axis=0)
    sv_norm = sv_norm.astype(dtype)
    ctx = tvm.context(target, 0)
    support_vectors = tvm.nd.array(support_vectors)
    dual_coef = tvm.nd.array(dual_coef)
    bias = tvm.nd.array(bias)
    gamma = clf._gamma
    coef0 = clf.coef0
    degree = clf.degree
    sv_norm = tvm.nd.array(sv_norm)
    n_support = tvm.nd.array(n_support)
    return support_vectors, dual_coef, bias, gamma, coef0, degree, sv_norm, n_support

def convert_linear_model_to_topi(model, data, dtype="float32", target="llvm"):
    """
    Convert linear model from sklearn 
    """
    c, b = parse_linear(model)
    X_tvm = te.placeholder((data.shape), name="X", dtype=dtype)
    C = te.placeholder((c.shape), name="C", dtype=dtype)
    B = te.placeholder((b.shape), name="B", dtype=dtype)
    with tvm.target.Target(target):
        tvm_func = func_name_parser(model)
        Y_tvm = tvm_func(X_tvm, C, B, dtype)
        #s = topi.x86.schedule_dense_pack([Y_tvm])
        #s = schedule_dense_pack([Y_tvm])
        s = schedule_classification([Y_tvm])
    ctx = tvm.context(target, 0)
    x_tvm = tvm.nd.array(data, ctx)
    y_tvm = tvm.nd.array(np.zeros(get_const_tuple(Y_tvm.shape), dtype=Y_tvm.dtype), ctx)
    func = tvm.build(s, [X_tvm, C, B, Y_tvm], target)
    data = (x_tvm, c, b, y_tvm)
    print(tvm.lower(s, (X_tvm, C, B, Y_tvm), simple_mode=True))
    return func, data

def convert_from_sklearn(model, data, dtype="float32", target="llvm"):
    """
    Input sklearn model
    Return model
    """
    model_name = str(model).split("(")[0]
    if model_name in ["LogisticRegression","LogisticRegressionCV","Perceptron","RidgeClassifier","RidgeClassifierCV","SGDClassifier","LinearRegression","Ridge","RidgeCV","SGDRegressor"]:
        func, data = convert_linear_model_to_topi(model, data, dtype=dtype, target=target)
    return func, data    
