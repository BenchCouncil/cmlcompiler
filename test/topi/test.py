"""test linear models"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.datasets import make_classification,make_multilabel_classification,make_regression
import numpy as np
from sklearn.model_selection import train_test_split
from cmlcompiler.utils.common_parser import parse_linear
from cmlcompiler.topi.linear import logistic_regression,logistic_regression_cv,ridge_classifier,ridge_classifier_cv,sgd_classifier,perceptron,linear_regression,ridge,ridge_cv,sgd_regressor
from cmlcompiler.topi.x86.linear import schedule_linear

def test_linear_model(sklearn_func, tvm_func, n_samples, n_features, n_classes=1, n_labels=1, dtype="float32", target="llvm"):
    """
    Testing linear models
    Input: sklearn function and corresponding tvm function
    Output: The result equals or not 
    """
    if sklearn_func in [LinearRegression, Ridge, RidgeCV, SGDRegressor]:
        X, y = make_regression(n_samples=n_samples, n_features=n_features)
    else:
        if(n_labels == 1):
            n_informative = int(n_features / 2)
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative)
        else:
            X, y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_labels=n_labels)
            
    # load dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # sklearn implements
    clf = sklearn_func()
    clf.fit(X_train, y_train)
    Y_test = clf.predict(X_test)
    X_shape = X_train.shape[1]
    c, b = parse_linear(clf)
    
    # tvm implements
    X_tvm = te.placeholder((X_test.shape), name="X", dtype=dtype)
    C = te.placeholder((c.shape), name="C", dtype=dtype)
    B = te.placeholder((b.shape), name="B", dtype=dtype)
    with tvm.target.Target(target):
        Y_tvm = tvm_func(X_tvm, C, B)
        s = schedule_linear([Y_tvm])
    ctx = tvm.context(target, 0)
    X_test = X_test.astype(np.float32)
    x_tvm = tvm.nd.array(X_test, ctx)
    y_tvm = tvm.nd.array(np.zeros(get_const_tuple(Y_tvm.shape), dtype=Y_tvm.dtype), ctx)
    func = tvm.build(s, [X_tvm, C, B, Y_tvm], target)
    func(x_tvm, c, b, y_tvm)
    
    # check
    y_np = np.array(Y_test)
    y_tvm = y_tvm.asnumpy()
    try:
        tvm.testing.assert_allclose(y_tvm, y_np, rtol=1e-4)
        print("pass")
    except Exception as e:
        print("error")
        print(e)
        for i in range(len(y_tvm)):
            if(abs(y_tvm[i] - y_np[i])>1e-4):
                print(y_tvm[i])
                print(y_np[i])

test_linear_model(LogisticRegression, logistic_regression, n_samples=1000, n_features=100, n_classes=10, n_labels=1)
