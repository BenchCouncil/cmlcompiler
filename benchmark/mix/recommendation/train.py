import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import log_loss
from benchmark.mix.recommendation.gbdt_feature import GradientBoostingFeatureGenerator
from benchmark.mix.recommendation.wide_deep import WideAndDeepModel
import torch
from torch.utils.data import DataLoader,TensorDataset
import tqdm
from sklearn.metrics import roc_auc_score

train_data_path = "avazu/train_data.csv"
gbdt_model_path = "avazu/gbdt_model"
cols = ['C1',
        'banner_pos', 
        'site_domain', 
        'site_id',
        'site_category',
        'app_id',
        'app_category', 
        'device_type', 
        'device_conn_type',
        'C14', 
        'C15',
        'C16']

cols_train = ['id', 'click']
cols_test  = ['id']
cols_train.extend(cols)
cols_test.extend(cols)

df_train = pd.read_csv(train_data_path)
X = df_train[cols].to_numpy()
y = df_train['click'].to_numpy()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)

"""
Train GBDT
"""
print("start training GBDT")
param = {  # init the hyperparams of GBDT
    'learning_rate': 0.2,
    'n_estimators': 100,  # number of trees here
    'max_depth': 3,  # set max_depth of a tree
    'subsample': 0.01, 
    'random_state': 1,
    'verbose': 0
    }
gbdt_model = GradientBoostingFeatureGenerator(**param)
gbdt_model.fit(X_train, y_train)
## store the pre-trained gbdt_model
pickle.dump(gbdt_model, open(gbdt_model_path, 'wb'))

X_feature = gbdt_model.transform(X)

"""
Train Wide and Deep model
"""

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields=fields.long()
        target=target.long()
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields=fields.long()
            target=target.long()
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)
    
class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train_wide_deep(epoch, X, y):
    device = torch.device("cpu")
    batch_size = 10
    learning_rate = 0.001
    weight_decay = 1e-6
    X = torch.from_numpy(X)
    print(X.shape[1])
    y = torch.from_numpy(y)
    dataset = TensorDataset(X, y)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    field_dims = [i for i in range(0, X.shape[1])]
    model = WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'avazu/wide_deep.pt')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')

train_wide_deep(100, X_feature, y_train)