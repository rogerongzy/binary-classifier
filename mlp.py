import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class dataset(Dataset):
    def __init__(self):
        self.X = torch.FloatTensor(np.array(scio.loadmat('data_train.mat')['data_train']))
        label_raw = np.array(scio.loadmat('label_train.mat')['label_train'])
        for idx in range(label_raw.shape[0]):
            if label_raw[idx, 0] == -1:
                label_raw[idx, 0] = 0
        self.y = torch.FloatTensor(label_raw)

        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


def data_config(ratio, bs):
    data = dataset()
    data_train, data_validation = random_split(data, [int(data.len * ratio), data.len - int(data.len * ratio)])
    train_loader = DataLoader(dataset=data_train, batch_size=bs, shuffle=True) # batch_size = 32
    validation_loader = DataLoader(dataset=data_validation, batch_size=128, shuffle=False)

    data_test = torch.FloatTensor(np.array(scio.loadmat('data_test.mat')['data_test']))
    test_loader = DataLoader(dataset=data_test, batch_size=128, shuffle=False)

    return train_loader, validation_loader, test_loader


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 33-10-8-1
        # 33-15-8-1
        self.layer1 = torch.nn.Linear(33, 15)
        torch.nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu') #
        self.act_func1 = torch.nn.ReLU()

        self.layer2 = torch.nn.Linear(15, 8)
        torch.nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu') #
        self.act_func2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Linear(8, 1)
        torch.nn.init.xavier_uniform_(self.layer3.weight) #
        self.act_func3 = torch.nn.Sigmoid() # softmax, ReLU


    def forward(self, x):
        x = self.layer1(x)
        x = self.act_func1(x)

        x = self.layer2(x)
        x = self.act_func2(x)

        x = self.layer3(x)
        x = self.act_func3(x)

        return x


def train_mlp(train_loader, lr):
    mlp = MLP()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr = lr)
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=lr, momentum=0.9) # lr=0.01, m=0.9
    loss_list = []

    for epoch in range(100):
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = mlp(X)
            loss = criterion(y_pred, y)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.item())

    # plt.plot(np.linspace(0,100,len(loss_list)),loss_list) # label='y1' + plt.legend()
    # # plt.xlim(0, 10)
    # plt.xlabel('epoch')
    # plt.ylabel('BCELoss')
    # plt.title('learning_date=0.01, batch_size=32')
    # # plt.savefig('learning_date=0.01,batch_size=32.png')
    # plt.show()

    return mlp, loss_list


def validate_mlp(mlp, validation_loader):
    num_correct = 0
    num_wrong = 0
    
    for batch, (X, y) in enumerate(validation_loader):
        for (X_, y_) in zip(X, y):
            y_pred = mlp(X_)
            y_pred = y_pred.data.item()
            
            if y_pred >= 0.5:
                class_prediction = 1
            else:
                class_prediction = 0

            if class_prediction == y_:
                num_correct = num_correct + 1
            else:
                num_wrong = num_wrong + 1

    return num_correct / (num_correct + num_wrong)

    
def get_score_list(ratio, bs):
    score_list = []
    for i in range(5000):
        train_loader, validation_loader = data_config(ratio, bs)
        mlp, loss_list = train_mlp(train_loader, 0.01)
        score_list.append(validate_mlp(mlp, validation_loader))
    return score_list


def print_score_hist():
    score_list = get_score_list(0.7, 32)
    plt.hist(score_list, range=(0.8,1), bins=39)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.savefig('Accuracy.png')
    plt.show()


def predict(mlp, test_loader):
    predict_list = []
    for batch, X in enumerate(test_loader):
        for X_ in X:
            y_pred = mlp(X_)
            y_pred = y_pred.data.item()

            if y_pred >= 0.5:
                class_prediction = 1
            else:
                class_prediction = -1
            
            predict_list.append(class_prediction)

    return predict_list


def svm_evaluate(mlp, validation_loader):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for batch, (X, y) in enumerate(validation_loader):
        for (X_, y_) in zip(X, y):
            y_pred = mlp(X_)
            y_pred = y_pred.data.item()
            
            if y_pred >= 0.5:
                if y_ == 1:
                    TP = TP + 1
                elif y_ == 0:
                    FP = FP + 1
            else:
                if y_ == 1:
                    FN = FN + 1
                elif y_ == 0:
                    TN = TN + 1               
    
    print('Accuracy: ' + str((TP+TN)/(TP+FP+FN+TN)))
    print('Precision: ' + str(TP/(TP+FP)))
    print('Recall: ' + str(TP/(TP+FN)))


if __name__ == '__main__':
    train_loader, validation_loader, test_loader = data_config(0.7, 32)
    mlp, loss_list = train_mlp(train_loader, 0.01)
    print(validate_mlp(mlp, validation_loader))
    print(predict(mlp, test_loader))

    svm_evaluate(mlp, validation_loader)

