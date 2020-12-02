# Machine Learning with Python
___

# 1. Training and testing split, cross validation and hiper-parameter tuning

## 1.1. Training and Testing Data

```Python
# Data preparation
X # pandas DataFrame (multiple columns)
y # pandas Series (one column)
# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
```

## 1.2. Cross validation

Main parameters:

- *cv*: determines the cross-validation splitting strategy (default value = 5). Literacy recommends between 5-10.
- *scoring*: scoring metric

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

### 2.1.2. Linear Regression - scikitlearn

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

- *n_neighbors*: number of neighbors k
- *weight*: 'uniform', 'distance'

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

- *max_depth*: number of splits
- *min_samples_leaf*: minimum number of observations per leaf. 
- *min_samples_split*: minimum number of observatons per leaf to subdivide. (Value 2 by default, be careful with overfitting.)

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

### 2.2.1. Logistic Regression

```Python
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
y_pred_prob = clf_lr.predict_proba(X_test) # probability values
```

### 2.2.2. K Nearest Neighbors (KNN)

Main parameters:

- *n_neighbors*: number of neighbors k
- *weight*: 'uniform', 'distance'

```Python
from sklearn.neighbors import KNeighborsClassifier
clf_kn = KNeighborsClassifier(n_neighbors=k)
clf_kn.fit(X_train, y_train)
y_pred = clf_kn.predict(X_test)
y_pred_prob = clf_kn.predict_proba(X_test) # probability values
```

### 2.2.3. Support Vector Machines (SVM)

Main parameters:

C: sum of error margins
kernel: linear, rbf(gamma=inverse of radius), poly(degree)

```Python
from sklearn.svm import SVC
clf_svm = SVC(kernel='linear', C=10)
clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)
y_pred_prob = clf_svm.predict_proba(X_test) # probability values
```

### 2.2.4. Decision Tree Classifier

Main parameters:

- *criterion*: Gini, entropy... (Gini by default)
- *max_depth*: number of splits
- *min_samples_leaf*: minimum number of observations per leaf. 
- *min_samples_split*: minimum number of observatons per leaf to subdivide. (Value 2 by default, be careful with overfitting.)

```Python
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=i, min_samples_leaf=j, random_state=99)
clf_dt.fit(X_train, y_train)
y_pred = clf_dt.predict(X_test)
y_pred_prob = clf_dt.predict_proba(X_test) # probability values
```

# 3. Unsupervised Learning Algorithms

# 4. Metrics

## 4.1. Regression metrics

- MAE is the easiest to understand, because it's the average error.
- MSE is more popular than MAE, because MSE punishes larger errors.
- RMSE is more popular than MSE, RMSE is interpretable in the y (dependant variable) units.

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

**R2 Score**

```Python
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```

**Correlation (between predictions and real value):**

```Python
# Direct Calculation
np.corrcoef(reg.predict(X_test),y_test)[0][1]
# Custom scorer
from sklearn.metrics import make_scorer
def corr(pred,y_test):
    return np.corrcoef(pred, y_test)[0][1]
# Put the scorer in cross_val_score
cross_val_score(reg, X, y, cv=5, scoring=make_scorer(corr))
```

**Bias (average of errors):**

```Python
# Direct Calculation
np.mean(reg.predict(X_test)-y_test)
# Custom Scorer
from sklearn.metrics import make_scorer
def bias(pred, y_test):
    return np.mean(pred-y_test)
# Put the scorer in cross_val_score
cross_val_score(reg, X, y, cv=5, scoring=make_scorer(bias))
```

## 4.2. Classification metrics

**Accuracy:**

% of correct predictions (tp + tf)

```Python
from sklearn import metrics 
metrics.accuracy_score(y_test, y_pred)
```

**Classification Report:**

Precision, Recall, F1 and Support

```Python
from sklearn import metrics 
print(metrics.classification_report(y_test, y_pred))
```

**Confusion Matrix:**

```Python
from sklearn import metrics
metrics.confusion_matrix(y_test, y_pred) 
```

```Python
from sklearn.metrics import confusion_matrix
import seaborn as sns
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",cmap="YlGnBu")
ax.set(xlabel='Predicted Values', ylabel='Actual Values', title='Confusion Matrix');
```

**Sensitivity (True Positive Rate or Recall):**

```Python
from sklearn import metrics
metrics.recall_score(y_test, y_pred)
```

**Specificity (True Negative Rate in 0/1 code):**

metrics.classification_report recall for 0

```Python
from sklearn import metrics
metrics.classification_report(y_test, y_pred)
```

**Precision:**

precision predicting positive instances

```Python
from sklearn import metrics
metrics.precision_score(y_test, y_pred)
```

**Recall:**

```Python
from sklearn import metrics
metrics.recall_score(y_test, y_pred)
```

**ROC curve:**

```Python
from sklearn.metrics import roc_curve
# We chose the target
target_pos = 1 # Or 0 for the other class
fp,tp,_ = roc_curve(y_test, y_pred_prob[:, target_pos])
plt.plot(fp, tp)
```

**AUC - Area under ROC curve:**

```Python
from sklearn.metrics import roc_curve, auc
fp,tp,_ = roc_curve(y_test, y_pred_prob[:,1])
auc(fp, tp)
```

5. Saving an Delivering a Model

```Python
import pickle
pickle.dump(reg_kn, open('model.pickle', 'wb'))
model_loaded = pickle.load(open('model.pickle', 'rb'))
```