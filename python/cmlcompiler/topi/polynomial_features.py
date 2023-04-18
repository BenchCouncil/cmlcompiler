"""polynomial features"""
import tvm
from tvm import te, topi, tir

def polynomial_features(x, degree=2, interaction_only=False):
    """
    Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree.
    
    For an input [a, b], in_features is n=2
    0-degree features are [1], count is 1
    1-degree features are [a, b], count is n
    2-degree features are [a^2, ab, b^2], count is n(n+1)/2
    total out features number is 1+n+n(n+1)/2
    do not support higher degree nowadays
    """
    # TODO: support higher degree
    n_samples, n = x.shape
    out = te.compute((n_samples, 1 + n), lambda i, j: te.if_then_else(j>0, x[i][j-1], 1))
    if interaction_only == False:
        for i in range(int(n)):
            features_tmp = n - i
            tmp = te.compute((n_samples, features_tmp), lambda k,j: x[k][i]*x[k][i+j])
            out = topi.concatenate([out, tmp], axis=1)
    elif interaction_only == True:
        for i in range(int(n)):
            features_tmp = n - i - 1
            tmp = te.compute((n_samples, features_tmp), lambda k,j: x[k][i]*x[k][i+j+1])
            out = topi.concatenate([out, tmp], axis=1)
    return out
