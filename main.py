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
classifier = SVC(kernel = "rbf" , random_state = 0 )
classifier.fit(X_train , Y_train)

Y_pred = classifier.predict(X_train)

from sklearn.metrics import accuracy_score , confusion_matrix
ac = accuracy_score(Y_train, Y_pred)
print(ac)
# print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1)) #concatenating predicted and actual values

