from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def split():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    return X_train, X_test

iris = load_iris()
X = iris.data
y = iris.target


result = []

for k in range(0,20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    accuracy = []
    for i in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors= i)
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_predict))
    result.append(accuracy)


for k in range(0,20):
    plt.plot(range(1,50),result[k])
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.show()
