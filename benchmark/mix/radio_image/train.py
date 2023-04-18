import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.transforms as transforms
from benchmark.mix.radio_image.simpleDNN import simpleDNN
import tqdm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from numpy import savetxt, genfromtxt

data_path = "CheXpert-v1.0-small"

"""
Only save radio figure and Pleural Effusion
"""
data = pd.read_csv(data_path + "/train.csv")
#Pleural Effusion
data = data[["Path", "Pleural Effusion"]]
data = data[data["Pleural Effusion"].isin([0, 1])]
data.to_csv(data_path + "/train_pleural.csv", index=False)

class CheXpertDataSet(Dataset):
    """
    CheXpert Dataset
    """
    def __init__(self, data_path, transform = None):
        """
        transform: optional transform for radio image
        """
        data_frame = pd.read_csv(data_path)
        image_names = data_frame["Path"].values.tolist()
        labels = data_frame["Pleural Effusion"].values.tolist() 
        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)
# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Tranform data
normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
transformList = []

transformList.append(transforms.Resize((imgtransCrop, imgtransCrop))) # 224
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

# Load dataset
dataset = CheXpertDataSet(data_path + "/train_pleural.csv", transformSequence)

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (X, target) in enumerate(tk0):
        target = target.reshape(-1, 1)
        y = model(X)
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
    ys, predicts = list(), list()
    with torch.no_grad():
        for X, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            y = model(X)
            y = [1 if i > 0.5 else 0 for i in y]
            ys.extend(y)
            predicts.extend(target.tolist())
    return accuracy_score(ys, predicts)

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

def train_simpleDNN(epoch, dataset):
    device = torch.device("cpu")
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-6
    # Use a small dataset to debug
    #used_dataset, unused_dataset = torch.utils.data.random_split(dataset, (int(len(dataset) * 0.01), (len(dataset) - int(len(dataset) * 0.01))))
    #dataset = used_dataset
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    #model = simpleDNN(1)
    model = torch.load("simpleDNN.pt") 
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'simpleDNN.pt')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')

train_simpleDNN(100, dataset)
#Save single sample for inference
data_loader = DataLoader(dataset, batch_size=1)
single_sample, single_y = next(iter(data_loader))
print(single_sample)
torch.save(single_sample, "single_sample.pt")

# Get feature embedding
simple_dnn = torch.load("simpleDNN.pt")
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
simple_dnn.fc3.register_forward_hook(get_activation('fc3'))
data_loader = DataLoader(dataset, batch_size=32)
y = list()
with open("data.csv", "a") as f_handle:
    with torch.no_grad():
        for X, target in data_loader:
            tmp = simple_dnn(X)
            y.extend(target.tolist())
            X_feature = activation["fc3"].numpy()
            #print(X_feature.shape)
            savetxt(f_handle, X_feature, delimiter=',')

savetxt("y.csv", y, delimiter=",")
X_feature = np.genfromtxt("data.csv", delimiter=",")
y = np.genfromtxt("y.csv", delimiter=",")
# Train random forest
forest = RandomForestClassifier(n_estimators=100, max_depth=4)
print("start training random forest")
forest.fit(X_feature, y)
pickle.dump(forest, open("forest", "wb"))