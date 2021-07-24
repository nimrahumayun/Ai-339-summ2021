import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn import svm

def cross_validate(estimator, train, validation):
    X_train = train[0]
    Y_train = train[1]
    X_val = validation[0]
    Y_val = validation[1]
    train_predictions = classifier.predict(X_train)
    train_accuracy = accuracy_score(train_predictions, Y_train)
    train_recall = recall_score(train_predictions, Y_train)
    train_precision = precision_score(train_predictions, Y_train)

    val_predictions = classifier.predict(X_val)
    val_accuracy = accuracy_score(val_predictions, Y_val)
    val_recall = recall_score(val_predictions, Y_val)
    val_precision = precision_score(val_predictions, Y_val)

    print('Model metrics')
    print('Accuracy  Train: %.2f, Validation: %.2f' % (train_accuracy, val_accuracy))
    print('Recall    Train: %.2f, Validation: %.2f' % (train_recall, val_recall))
    print('Precision Train: %.2f, Validation: %.2f' % (train_precision, val_precision))

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
test_ids = test_df['PassengerId'].values
train_df=train_df.drop("PassengerId",axis=1)
train_df=train_df.drop("Name",axis=1)
train_df=train_df.drop("Ticket",axis=1)
train_df=train_df.drop("Cabin",axis=1)
test_df=test_df.drop("Name",axis=1)
test_df=test_df.drop("Ticket",axis=1)
test_df=test_df.drop("Cabin",axis=1)
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()
common_value = 'S'
train_df["Embarked"] = train_df["Embarked"].fillna(common_value)
test_df = test_df.fillna(test_df['Fare'].mean())
le = LabelEncoder()
train_df["Sex"]= le.fit_transform(train_df["Sex"])
le = LabelEncoder()
test_df["Sex"]= le.fit_transform(test_df["Sex"])
le = LabelEncoder()
train_df["Embarked"]= le.fit_transform(train_df["Embarked"])
le = LabelEncoder()
test_df["Embarked"]= le.fit_transform(test_df["Embarked"])
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


target = 'Survived'
labels = train_df[target].values
X_train, X_val, Y_train, Y_val = train_test_split(train_df, labels, test_size=0.2, random_state=1)

classifier = linear_model.LogisticRegression()
classifier.fit(X_train, Y_train)
print("Logistic Regression")
cross_validate(classifier, (X_train, Y_train), (X_val, Y_val))
test_df.fillna(test_df.mean(), inplace=True)
test_predictions = classifier.predict(test_df)
submission = pd.DataFrame({'PassengerId': test_ids})
submission['Survived'] = test_predictions.astype('int')
submission.to_csv('LRF3.csv', index=False)
classifier = svm.SVC()
classifier.fit(X_train, Y_train)
print("SVM")
cross_validate(classifier, (X_train, Y_train), (X_val, Y_val))
test_df.fillna(test_df.mean(), inplace=True)
test_predictions = classifier.predict(test_df)
submission = pd.DataFrame({'PassengerId': test_ids})
submission['Survived'] = test_predictions.astype('int')
submission.to_csv('SVMF3.csv', index=False)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)
print("KNN")
cross_validate(classifier, (X_train, Y_train), (X_val, Y_val))
test_df.fillna(test_df.mean(), inplace=True)
test_predictions = classifier.predict(test_df)
submission = pd.DataFrame({'PassengerId': test_ids})
submission['Survived'] = test_predictions.astype('int')
submission.to_csv('KNNF3.csv', index=False)
