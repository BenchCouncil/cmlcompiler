"""test svm models"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,SVC,SVR
from sklearn.datasets import make_classification,make_multilabel_classification,make_regression
import numpy as np
from sklearn.model_selection import train_test_split
from hummingbird.ml import convert
from cmlcompiler.model import build_model
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
import os
from cmlcompiler.utils.supported_ops import svm_clf,svm_reg
import tvm.testing

os.environ["TVM_BACKTRACE"] = "1"

def test_linear_svm(sklearn_func, n_samples, n_features, n_classes=1, n_labels=1, dtype="float32", target="llvm"):
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
    model = build_model(clf, data.shape)
    y_tvm = model.run(data)
    # check
    y_np = np.array(Y_test)
    try:
        tvm.testing.assert_allclose(y_tvm, y_np, rtol=1e-4)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

def test_nolinear_svm(sklearn_func, kernel, n_samples, n_features, n_classes=1, n_labels=1, dtype="float32", target="llvm"):
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
    # tvm implements
    data = X_test
    model = build_model(clf, data.shape)
    y_tvm = model.run(data)
    # check
    y_np = np.array(Y_test)
    try:
        tvm.testing.assert_allclose(y_tvm, y_np, rtol=1e-3)
        print("pass")
    except Exception as e:
        print("error")
        print(e)
test_linear_svm(LinearSVR, n_samples=1010, n_features=100)
test_linear_svm(LinearSVC, n_samples=1010, n_features=100, n_classes=10, n_labels=1)
test_linear_svm(LinearSVC, n_samples=1010, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(SVR, kernel="linear", n_samples=1010, n_features=100, n_classes=10, n_labels=1)
test_nolinear_svm(NuSVR, kernel="linear", n_samples=1010, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(SVR, kernel="sigmoid", n_samples=1010, n_features=100, n_classes=10, n_labels=1)
test_nolinear_svm(NuSVR, kernel="sigmoid", n_samples=1010, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(SVR, kernel="poly", n_samples=1010, n_features=100, n_classes=10, n_labels=1)
test_nolinear_svm(NuSVR, kernel="poly", n_samples=1010, n_features=100, n_classes=2, n_labels=1)
test_nolinear_svm(SVR, kernel="rbf", n_samples=1010, n_features=100, n_classes=10, n_labels=1)
test_nolinear_svm(NuSVR, kernel="rbf", n_samples=1010, n_features=100, n_classes=2, n_labels=1)
#test_nolinear_svm(SVC, kernel="linear", n_samples=100, n_features=20, n_classes=4, n_labels=1)
#test_nolinear_svm(SVC, kernel="sigmoid", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
#test_nolinear_svm(SVC, kernel="poly", n_samples=1000, n_features=100, n_classes=2, n_labels=1)
#test_nolinear_svm(SVC, kernel="linear", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
#test_nolinear_svm(SVC, kernel="sigmoid", n_samples=40, n_features=100, n_classes=3, n_labels=1)
#test_nolinear_svm(SVC, kernel="poly", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
#test_nolinear_svm(SVC, kernel="rbf", n_samples=1000, n_features=100, n_classes=10, n_labels=1)
#test_nolinear_svm(SVC, kernel="rbf", n_samples=100, n_features=10, n_classes=2, n_labels=1)
