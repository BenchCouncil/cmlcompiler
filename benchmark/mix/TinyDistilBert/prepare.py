import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import transformers
import pickle
import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv("./data/train.tsv", delimiter="\t", header=None)
test_data = pd.read_csv("./data/train.tsv", delimiter="\t", header=None)

#pretrained_weights = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
#pretrained_weights = "sgugger/tiny-distilbert-classification"
#pretrained_weights = "sshleifer/tiny-distilroberta-base"
pretrained_weights = "prajjwal1/bert-tiny"
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_weights)
model = transformers.AutoModel.from_pretrained(pretrained_weights)

def prepare_data(data):
    tokenized = data[0].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    return padded, attention_mask

train_padded, train_mask = prepare_data(train_data)
test_padded, test_mask = prepare_data(test_data)
np.savetxt("test_padded.csv", test_padded, delimiter=",")
np.savetxt("test_mask.csv", test_mask, delimiter=",")

def run_distilbert(padded, mask):
    """
    Run DistilBert, using pretrained model
    """
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    train_features = last_hidden_states[0][:, 0, :].numpy()
    return train_features

train_features = run_distilbert(train_padded, train_mask)
test_features = run_distilbert(test_padded, test_mask)
train_labels = train_data[1]
test_labels = test_data[1]

def train_lr(train_features, train_labels):
    """
    Train logistic regression 
    """
    parameters = {"C":np.linspace(0.0001, 100, 20)}
    grid_search = GridSearchCV(LogisticRegression(), parameters)
    grid_search.fit(train_features, train_labels)
    lr_clf = LogisticRegression(C=grid_search.best_params_["C"])
    lr_clf.fit(train_features, train_labels)
    return lr_clf

def train_tree(train_features, train_labels):
    """
    Train decision tree
    """
    tree_clf = DecisionTreeClassifier(max_depth=4)
    tree_clf.fit(train_features, train_labels)
    return tree_clf

def train_forest(train_features, train_labels):
    """
    Train Random Forest
    """
    forest_clf = RandomForestClassifier(max_depth=4)
    forest_clf.fit(train_features, train_labels)
    return forest_clf

def train_SVC(train_features, train_labels):
    """
    Train decision tree
    """
    clf = SVC(kernel="linear")
    clf.fit(train_features, train_labels)
    return clf

lr_clf = train_lr(train_features, train_labels)
pickle.dump(lr_clf, open("lr_clf.sav", "wb"))
#print(lr_clf.score(test_features, test_labels))
#tree_clf = train_tree(train_features, train_labels)
#pickle.dump(tree_clf, open("tree_clf.sav", "wb"))
#print(tree_clf.score(test_features, test_labels))
#forest_clf = train_forest(train_features, train_labels)
#pickle.dump(forest_clf, open("forest_clf.sav", "wb"))
#print(forest_clf.score(test_features, test_labels))
#clf = train_SVC(train_features, train_labels)
#pickle.dump(clf, open("svc_clf.sav", "wb"))
#print(clf.score(test_features, test_labels))
