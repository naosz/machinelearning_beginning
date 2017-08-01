from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris = load_iris()

X = iris.data
y = iris.target

#define the model
knn = KNeighborsClassifier(n_neighbors=1)

#define logistic regression
logreg = LogisticRegression()

#train kNN
knn.fit(X,y)

#create new test list
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]

#predict 2nd sample
second_test = knn.predict(X_new)
print("Knn Prediciton",second_test)

#train logreg
logreg.fit(X,y)

y_pred = logreg.predict(X)
print(len(y_pred))

second_test_logreg = logreg.predict(X_new)
print(second_test_logreg)

#compute classification accuracy for logReg model
print(metrics.accuracy_score(y,y_pred))
