# Machine Learning with Python
___

# 1. Training and Testing Data, Cross validation and hiper parameter tuning

## 1.1. Training and Testing Data

```Python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
```

## 1.2. Cross validation

Main parameters:

- cv: determines the cross-validation splitting strategy (default value = 5). Literacy recommends between 5-10.
- scoring: scoring metric

```Python
from sklearn.model_selection import cross_val_score
clf_kn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(clf_kn, X, y, cv=5, scoring='accuracy').mean()
```

## 1.3. GridSearchCV

## 1.4. RandomizedSearchCV

# 2. Supervised Learning Algorithms

## 2.1. Regression

### 2.1.1. Linear Regression - statsmodel

```Python
# Load the library
import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales~TV+Radio', data=training).fit
lm.pvalues
lm.summary()
sales_pred = lm.predict(testing)
```

### 2.1.2. Linear Regression - sklearn

```Python
# Load the library
from sklearn.linear_model import LinearRegression
# Create an instance of the model
reg_lin = LinearRegression()
# Fit the regressor
reg_lin.fit(X_train, y_train)
# Coeficients
reg_lin.coef_, reg_lin.intercept_
# Predictions
y_pred = reg_lin.predict(X_test)

# Graphics
x_hip = np.linspace(start, stop, sample)
y_hip = reg_lin.predict(pd.DataFrame(x_hip))
plt.scatter(X, y, c='lightgreen', alpha=0.2)
plt.plot(x_hip, y_hip, c='red');
```

### 2.1.3. K Nearest Neighbors (KNN)

Main parameters:

- Number of neighbors $k$

```Python
from sklearn.neighbors import KNeighborsRegressor
reg_kn = KNeighborsRegressor(n_neighbors=k)
reg_kn.fit(X_train, y_train)
y_pred = reg_kn.predict(X_test)

# Graphics
x_hip = np.linspace(start, stop, sample)
y_hip = reg_kn.predict(pd.DataFrame(x_hip))
plt.scatter(X, y, c='lightgreen', alpha=0.2)
plt.plot(x_hip, y_hip, c='red');
```

### 2.1.4. Decision Trees

Main parameters:

- max_depth: number of splits
- min_samples_leaf: minimum number of observations per leaf. 
- min_samples_split: minimum number of observatons per leaf to subdivide. (Value 2 by default, be careful with overfitting.)

```Python
from sklearn.tree import DecisionTreeRegressor
reg_dt = DecisionTreeRegressor(max_depth=i, min_sample_leaf=j)
reg_dt.fit(X_train, y_train)
y_pred = reg_dt.predict(X_test)

# Graphics
x_hip = np.linspace(start, stop, sample)
y_hip = reg_dt.predict(pd.DataFrame(x_hip))
plt.scatter(X, y, c='lightgreen', alpha=0.2)
plt.plot(x_hip, y_hip, c='red');
```

## 2.2. Classification

# 3. Unsupervised Learning Algorithms

# 4. Metrics

## 4.1. Regression metrics

- MAE is the easiest to understand, because it's the average error.
- MSE is more popular than MAE, because MSE "punishes" larger errors.
- RMSE is more popular than MSE, RMSE is interpretable in the "y" units.

**MAE - Mean Absolute Error**

```Python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)
```

**MAPE - Mean Absolute Percentage Error**

```Python
import numpy as np
np.mean(np.abs(y_test-y_pred)/y_test)
```

**MSE - Mean Squared Error**

```Python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)
```

**RMSE - Root Mean Squared Error**

```Python
import numpy as np
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))
```

**$R^2$ Score**

```Python
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```

5. Saving an Delivering a Model

```Python
import pickle
pickle.dump(reg_kn, open('model.pickle', 'wb'))
model_loaded = pickle.load(open('model.pickle', 'rb'))
```