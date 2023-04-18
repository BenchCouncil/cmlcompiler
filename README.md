# CMLCompiler 

## Installation
1. Install tvm, following https://tvm.apache.org/docs/install/index.html

2. 
`
git clone https://github.com/warmth1905/cmlcompiler 
`

3. Add 

`
export CMLCOMPILER_HOME=/path/of/cmlcompiler
`

`
export PYTHONPATH=$CMLCOMPILER_HOME/python:${PYTHONPATH}
`

to bashrc

## Usage

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from cmlcompiler.model import build_model
X = np.random.rand(1000, 100)
y = np.random.randint(2, size=1000)
clf = LogisticRegression()
clf.fit(X, y)
batch_size = 1000
target = "llvm"
model = build_model(clf, X.shape, batch_size=batch_size, target=target)
out = model.run(X)
```