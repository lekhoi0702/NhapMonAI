import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


dataSet = pd.read_csv('Iris.csv')
dataSet.isna().sum()
dataSet.duplicated().sum()



# Tách ra nhãn và thuộc tính
X = dataSet.drop('Species', axis=1)
y = dataSet['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
for i in range(len(X_test)):
    print(f"Name:{i+1}: {X_test[i]} --> Predict: {y_pred[i]}")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
