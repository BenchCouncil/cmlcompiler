"""test svm models"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,SVC,SVR
from sklearn.datasets import make_classification,make_multilabel_classification,make_regression
import numpy as np
from sklearn.model_selection import train_test_split
from common_parser import parse_linear,parse_svm
from cmlcompiler.topi.svm import linear_svc,linear_svr,svc,svr,nu_svc,nu_svr
from hummingbird.ml import convert
from cmlcompiler.model import build_model

def test_linear_svm(sklearn_func, tvm_func, n_samples, n_features, n_classes=1, n_labels=1, dtype="float32", target="llvm"):
    """
    Testing linear SVM models
    Input: sklearn function and corresponding tvm function
    Output: The result equals or not 
    """
    if sklearn_func==LinearSVR:
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
    data = X_test
    model = build_model(clf, data.shape, out_dtype="int")
    y_tvm = model.run(data).asnumpy()
    """
    c, b = parse_linear(clf)
    
    # tvm implements
    X_tvm = te.placeholder((X_test.shape), name="X", dtype=dtype)
    C = te.placeholder((c.shape), name="C", dtype=dtype)
    B = te.placeholder((b.shape), name="B", dtype=dtype)
    Y_tvm = tvm_func(X_tvm, C, B)
    s = te.create_schedule(Y_tvm.op)
    ctx = tvm.context(target, 0)
    X_test = X_test.astype(np.float32)
    x_tvm = tvm.nd.array(X_test, ctx)
    y_tvm = tvm.nd.array(np.zeros(get_const_tuple(Y_tvm.shape), dtype=Y_tvm.dtype), ctx)
    func = tvm.build(s, [X_tvm, C, B, Y_tvm], target)
    func(x_tvm, c, b, y_tvm)
    """
    # check
    y_np = np.array(Y_test)
    y_tvm = y_tvm.asnumpy().flatten()
    try:
        tvm.testing.assert_allclose(y_tvm, y_np, rtol=1e-4)
        print("pass")
    except Exception as e:
        print("error")
        print(e)
        """
        for i in range(len(y_tvm)):
            if(abs(y_tvm[i] - y_np[i])>1e-4):
                print(y_tvm[i])
                print(y_np[i])
        """
def test_nolinear_svm(sklearn_func, tvm_func, kernel, n_samples, n_features, n_classes=1, n_labels=1, dtype="float32", target="llvm"):
    """
    Testing nolinear SVM models
    Input: sklearn function and corresponding tvm function
    Output: The result equals or not 
    """
    if sklearn_func in [NuSVR,SVR]:
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
    clf = sklearn_func(kernel=kernel)
    clf.fit(X_train, y_train)
    Y_test = clf.predict(X_test)
    support_vectors, dual_coef, bias, gamma, coef0, degree, sv_norm, n_support = parse_svm(clf)
    
    # tvm implements
    X_tvm = te.placeholder((X_test.shape), name="X", dtype=dtype)
    SV = te.placeholder((support_vectors.shape), dtype=dtype)
    DC = te.placeholder((dual_coef.shape), dtype=dtype)
    B = te.placeholder((bias.shape), dtype=dtype)
    SV_NORM = te.placeholder((sv_norm.shape), dtype=dtype)
    N_SUPPORT = te.placeholder((n_support.shape), dtype=dtype)
    Y_tvm = tvm_func(X_tvm, kernel, gamma, coef0, degree, SV_NORM, SV, DC, B, N_SUPPORT)
    s = te.create_schedule(Y_tvm.op)
    ctx = tvm.context(target, 0)
    X_test = X_test.astype(dtype)
    x_tvm = tvm.nd.array(X_test, ctx)
    y_tvm = tvm.nd.array(np.zeros(get_const_tuple(Y_tvm.shape), dtype=Y_tvm.dtype), ctx)
    func = tvm.build(s, [X_tvm, SV_NORM, SV, DC, B, N_SUPPORT, Y_tvm], target)
    func(x_tvm, sv_norm, support_vectors, dual_coef, bias, n_support, y_tvm)
    

    data = X_test
    model = build_model(clf, data.shape, out_dtype="float32")
    y_relay = model.run(data).asnumpy()
    y_tvm = y_tvm.asnumpy()
    # check
    try:
        tvm.testing.assert_allclose(y_tvm, y_relay, rtol=1e-3)
        print("pass")
    except Exception as e:
        print("error")
        print(e)
#test_linear_svm(LinearSVR, linear_svr, n_samples=1000, n_features=100)
#test_linear_svm(LinearSVC, linear_svc, n_samples=1000, n_features=100, n_classes=10, n_labels=1)
#test_linear_svm(LinearSVC, linear_svc, n_samples=1000, n_features=100, n_classes=2, n_labels=1)
"""
test_nolinear_svm(SVR, svr, kernel="linear", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
test_nolinear_svm(SVR, svr, kernel="sigmoid", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
test_nolinear_svm(SVR, svr, kernel="poly", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
test_nolinear_svm(SVR, svr, kernel="rbf", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
test_nolinear_svm(NuSVR, nu_svr, kernel="linear", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(NuSVR, nu_svr, kernel="sigmoid", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(NuSVR, nu_svr, kernel="poly", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
"""
test_nolinear_svm(NuSVR, nu_svr, kernel="rbf", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
"""
test_nolinear_svm(SVC, svc, kernel="linear", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(SVC, svc, kernel="sigmoid", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(SVC, svc, kernel="poly", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(SVC, svc, kernel="rbf", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(SVC, svc, kernel="linear", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
"""
#test_nolinear_svm(SVC, svc, kernel="sigmoid", n_samples=40, n_features=100, n_classes=3, n_labels=1)
#test_nolinear_svm(SVC, svc, kernel="poly", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
#test_nolinear_svm(SVC, svc, kernel="rbf", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
