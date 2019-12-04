# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

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
    
    # one-hot encode embarkation port...
    columnTransformer = ColumnTransformer(
        [('passthrough', 'passthrough', [0,1,2,3,4,5,6]),
         ('one-hot', OneHotEncoder(), [7]),
        ])
    X = columnTransformer.fit_transform(X)
    
    # TODO: one-hot ticket class
    
    # TODO: feature normalization
    
    return X

# extract our ground truth...
y = train_data.values[:,1]
y = y.astype('int')  # y datatype was 'object'

train_data = train_data.drop(['Survived'], axis=1)

X = pre_process(train_data)
X_test = pre_process(test_data)

lr = LogisticRegression(random_state=0, solver='liblinear', max_iter=500).fit(X, y)
predictions = lr.predict(X_test)

predictions = np.concatenate([X_test[:,0].reshape(-1,1), predictions.reshape(-1,1)], axis=1)

out = pd.DataFrame(data=predictions, columns=['PassengerId','Survived'])
print(out)
out.to_csv('out.csv', index = False)