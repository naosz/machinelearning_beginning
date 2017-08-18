#learning to us regression modell
#first thing i realized is i dont know shit about regression, time for youtube
import  matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from cycler import cycler

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]

#this is a retarded way to split the sample into test and train, so i made it with a function i learned form knn.
'''
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
'''



for i in range(0,10):
    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes.target, test_size=0.3)

    regr = linear_model.LinearRegression()

    regr.fit(diabetes_X_train, diabetes_y_train)

    print("Coefficient: \n", regr.coef_)
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    plt.scatter(diabetes_X_test, diabetes_y_test)
    plt.plot(diabetes_X_test, regr.predict(diabetes_X_test))

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))
plt.xticks(())
plt.yticks(())
plt.show()

plt.savefig('display.png')

