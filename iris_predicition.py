from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris = load_iris()

X = iris.data
y = iris.target

#define the model
knn = KNeighborsClassifier(n_neighbors=2)

#define logistic regression
logreg = LogisticRegression()

#train kNN
knn.fit(X,y)

#create new test list
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]

#predict 2nd sample
knn_pred = knn.predict(X)

#train logreg
logreg.fit(X,y)

log_pred = logreg.predict(X)
#print(len(log_pred))

#compute classification accuracy for logReg model
print("knn",metrics.accuracy_score(y,knn_pred))
print("log",metrics.accuracy_score(y,log_pred))

for i in range(5):
    print(i)

