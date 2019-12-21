# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import array as array
import pickle

# Importing the dataset
dataset = pd.read_csv('attendance_example.csv')
X = dataset.iloc[:, 1:6]
y = dataset.iloc[:, 6].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_d = LabelEncoder()
X['day'] = labelencoder_d.fit_transform(X['day'])
labelencoder_r = LabelEncoder()
X['religion'] = labelencoder_r.fit_transform(X['religion'])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

test = np.array([[98, 2, 1, 4, 1]])
test.reshape(1, -1)
res = classifier.predict(test)

def calc_d():
    print("Hello")
