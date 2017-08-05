import  matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

result = []

for k in range(0,30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    accuracy = []
    for i in range(1, 35):
        knn = KNeighborsClassifier(n_neighbors= i)
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_predict))
    result.append(accuracy)

averg = numpy.mean(result,axis=0)

ax1 = plt.subplot(211)

for k in range(0,30):
    plt.plot(range(1,35),result[k])
plt.xlabel("K values")
plt.ylabel("Accuracy")

plt.subplot(212,sharey=ax1)

plt.plot(range(1,35),averg)
plt.xlabel("K values")
plt.ylabel("Accuracy")

plt.savefig('display.png')

top_knn_value = numpy.amax(averg)
print("Top average:",top_knn_value)
print("K value:",numpy.argmax(averg)+1)
print(averg)
