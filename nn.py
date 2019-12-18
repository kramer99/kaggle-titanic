import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# preprocessing is all the same as for the linear regression model, I would extract it out, but this has to run in a notebook

def pre_process(data):
    # remove extraneous columns: Name, TicketNo, Cabin (Cabin could potentially be useful if it could be mapped to a location on the ship)
    data = data.drop(['Name','Ticket','Cabin'], axis=1)
    
    # replace null ages, fares with means rather than simply delete the rows, as there is not that much data...
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    
    # two records have null embarkation port, just going to replace with the most common port: 'Southhampton'
    data['Embarked'].fillna('S', inplace=True)
    
    X = data.values
    
    # encode gender...
    labelencoder = LabelEncoder()
    X[:, 2] = labelencoder.fit_transform(X[:, 2])
    
    # remove passengerId. one-hot encode ticket class, embarkation port...
    columnTransformer = ColumnTransformer(
        [('passthrough', 'passthrough', [2,3,4,5,6]),
         ('one-hot', OneHotEncoder(), [1,7]),
        ])
    X = columnTransformer.fit_transform(X)
    
    # feature normalization
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    
    return X

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(11, 6) # after pre-processing, there are 11 features
        self.fc2 = nn.Linear(6, 3)  # 2 hidden layers should enable more complex curved decision boundary to form.
        self.fc3 = nn.Linear(3, 1)  # binary output unit
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x))) # apply RELU activation function to outputs of first layer, with 15% dropout chance
        x = F.relu(self.fc2(x)) 
        x = torch.sigmoid(self.fc3(x)) # output layer 
        return x

def predict(model, X):
    input = torch.from_numpy(X).float()
    output = model(input)
    predictions = np.where(output > .5, 1, 0).reshape(-1)
    return predictions

def accuracy(predictions, y):
    same = np.where(predictions != y, 0, 1)
    return np.sum(same) / y.shape[0]

def train(model, X_train, y_train, alpha, regularization):
    input_train = torch.from_numpy(X_train).float()
    target = torch.from_numpy(y_train).float()
    target = target.view(-1,1)    # reshape y matrix to stop pytorch freaking out
    optimizer = optim.Adam(model.parameters(), lr=alpha, weight_decay=regularization)
    
    for epoch in range(1000):
        output = model(input_train)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
        loss.backward()
        optimizer.step()    # update weights and bias parameters on the model
        if epoch % 100 == 0:
            print(epoch, ': ', loss)

def local_dev():
    # split some of the training data off as test data because the Kaggle supplied test data has no y value to verify on...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    # try various combinations of hyperparameters...
    #for alpha in [1, 0.6, 0.3, 0.1, 0.06, 0.03, 0.01]:
    #    for r in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    #        print('alpha: ', alpha, ' r: ', r)
    #        model = Model()
    #        train(model, X_train, y_train, alpha, r)

    model = Model()
    train(model, X_train, y_train, 0.06, 0.00001)   # these alpha and reg values were selected after experimenting with the commented section above
    predictions_train = predict(model, X_train)
    print('Accuracy on training data:', accuracy(predictions_train, y_train))
    predictions_test = predict(model, X_test)
    print('Accuracy on test data:', accuracy(predictions_test, y_test))

def submission():
    X_test_kaggle = pre_process(kaggle_test_data)
    kaggle_test_passenger_ids = kaggle_test_data['PassengerId'].values
    
    model = Model()
    train(model, X, y, 0.06, 0.00001)   # these alpha and reg values were selected after experimenting with the commented section above
    predictions = predict(model, X_test_kaggle)
    predictions = np.concatenate([kaggle_test_passenger_ids.reshape(-1,1), predictions.reshape(-1,1)], axis=1)
    out = pd.DataFrame(data=predictions, columns=['PassengerId','Survived'])
    out.to_csv('out.csv', index = False)
    

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
kaggle_test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# extract our ground truth...
y = train_data.values[:,1]
y = y.astype('int')  # y datatype was 'object'
train_data = train_data.drop(['Survived'], axis=1)

X = pre_process(train_data)

#torch.manual_seed(0)
#local_dev()
submission()