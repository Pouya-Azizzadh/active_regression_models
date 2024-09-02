import numpy as np
import pandas as pd
from statistics import mean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  # Changed to LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score

from sklearn.datasets import make_regression

# Generate a dataset with 1000 samples, 5 features, and a noise level of 10
# X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)


def split(x, y, train_size, test_size):
    x_train, x_pool, y_train, y_pool = train_test_split(
        x, y, train_size=train_size)
    unlabel, x_test, label, y_test = train_test_split(
        x_pool, y_pool, test_size=test_size)
    return x_train, y_train, x_test, y_test, unlabel, label


class FuzzySet:
    def __init__(self, type, *params):
        self.type = type
        self.params = params

    def membership(self, x):

        if self.type == 'triangular':
            a, b, c = self.params

            if x <= a:
                return 0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b < x <= c:
                return (c - x) / (c - b)
            else:
                return 0
        elif self.type == 'trapezoidal':
            a, b, c, d = self.params
            if x <= a:
                return 0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b <= x <= c:
                return 1
            elif c < x <= d:
                return (d - x) / (d - c)
            else:
                return 0
        else:
            raise ValueError("Invalid fuzzy set type")




from sklearn.linear_model import Ridge

# Modify active_learning function to use Ridge Regression
def active_learning(max_iter, train_len, test_len, MODEL, X, y):
    model = MODEL
    
    ac1 = [] 
    y_pred = []
    for i in range(max_iter):
        print(f"iter:{i} from {max_iter} iterations")
        x_train, y_train, x_test, y_test, unlabel, label = split(X, y, train_len,test_len)

        for i in range(unlabel.shape[0]):
            model.fit(x_train, y_train)
            y_pred_temp = model.predict(unlabel)

            p = FuzzySet('triangular', np.min(y_pred_temp), np.mean(y_pred_temp), np.max(y_pred_temp)).membership(
                y_pred_temp[i - 1])
            uncrt_pt_ind = []
        if unlabel.shape[0] > 0:
            for i in range(unlabel.shape[0]):
                if (y_pred_temp[i] >= p and y_pred_temp[i] <= 1 - p):
                    uncrt_pt_ind.append(i)
            x_new_train = np.append(unlabel[uncrt_pt_ind, :], x_train, axis=0)
            y_new_train = np.append(label[uncrt_pt_ind], y_train)
            unlabel = np.delete(unlabel, uncrt_pt_ind, axis=0)
            label = np.delete(label, uncrt_pt_ind)
        uncrt_pt_ind = np.where((y_pred_temp >= p) & (y_pred_temp <= 1 - p))[0]

        model_2 = MODEL
        model_2.fit(x_new_train, y_new_train)
        y_pred = model_2.predict(x_test)
        y_pred_all = model_2.predict(X)
        mse_all = accuracy_score(y, y_pred_all)

        mse = accuracy_score(y_test, y_pred)
        print("accuracy_score:", mse)
        ac1.append(mse)

        train_size = x_train.shape[0]/X.shape[0]

    y_pred_train=model.predict(x_new_train)
    print("r2 by active model :", accuracy_score(y,y_pred_all))
    print("r2 test by active model :", accuracy_score(y_test,y_pred))
 
    print("r2 train by active model :", accuracy_score(y_new_train,y_pred_train))

    return y,y_pred_all,y_test,y_pred,x_new_train, y_new_train,y_pred_train

# y_pred,y_pred_all=active_learning(200, 0.01,0.3, LinearRegression(),X, y )