from pprint import pprint
import pandas as pd
import numpy as np

dataset = pd.read_csv("parkinsons.csv")
# print(dataset)

# print(dataset.info())

# X = dataset.iloc[1:,:-1].values
# Y = dataset.iloc[:-1].values

# print(X)
# # print(Y)


# print(dataset['status'].value_counts())

# print(dataset.groupby('status').mean())

X = dataset.drop(columns=['name' , 'status'], axis=1)
# print(X)
Y = dataset['status']
# print(Y)


from sklearn.model_selection import train_test_split
X_test , X_train , Y_test , Y_train = train_test_split(X , Y , test_size=0.2 , random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train  = sc.fit_transform(X_test)
X_test = sc.transform(X_test)

#training model

from sklearn.svm import SVC
classifier = SVC(kernel = "rbf")
classifier.fit(X_train , Y_train)

y_pred = classifier.predict(X_train)

from sklearn.metrics import accuracy_score , confusion_matrix
ac = accuracy_score()

