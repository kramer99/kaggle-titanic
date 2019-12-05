import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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

def local_dev():
    # split some of the training data off as test data because the Kaggle supplied test data has no y value to verify on...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    lr = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000).fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    print(score)

def submission():
    X_test_kaggle = pre_process(kaggle_test_data)
    kaggle_test_passenger_ids = kaggle_test_data['PassengerId'].values
    
    lr = LogisticRegression(random_state=0, solver='liblinear', max_iter=1000).fit(X, y)
    predictions = lr.predict(X_test_kaggle)
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

local_dev()
#submission()
