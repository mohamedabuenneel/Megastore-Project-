import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn as kt
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, QuantileRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import linear_model
from sklearn import metrics
import seaborn as sns
import sklearn
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from MyLabelEncoder import *


def Feature_Encoder_train(df, X, lbl):
    df[X] = lbl.fit_transform(list(df[X]))
    return df[X]


def Feature_Encoder_test(df, X, lbl):
    df[X] = lbl.transform(list(df[X]))
    return df[X]


data = pd.read_csv(
    "D:\\Programming\\Projects\\coding\\fcis_projects\\ML_Project\\Data\\megastore-regression-dataset.csv")
# Checking Dataframe
data.head()
data.info()

# check null
print(data.isna().sum())
data = data.dropna(axis=0, how='any', subset=None, inplace=False)

X = data.drop('Profit', axis=1)
Y = data['Profit']

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=47)

# ------------------------------------------------------------------------------------------------------------
# feature engineering
# calculate the time to deliver column

X_train['date1Train'] = pd.to_datetime(X_train['Ship Date'])
X_train['date2Train'] = pd.to_datetime(X_train['Order Date'])

X_test['date1Test'] = pd.to_datetime(X_test['Ship Date'])
X_test['date2Test'] = pd.to_datetime(X_test['Order Date'])

delta = X_train['date1Train'] - X_train['date2Train']
delta_days = delta.dt.days
X_train['days to deliver'] = delta_days

delta = X_test['date1Test'] - X_test['date2Test']
delta_days = delta.dt.days
X_test['days to deliver'] = delta_days

X_train.drop('date1Train', axis=1, inplace=True)
X_train.drop('date2Train', axis=1, inplace=True)
X_test.drop('date1Test', axis=1, inplace=True)
X_test.drop('date2Test', axis=1, inplace=True)
# ------------------------------------------------------------------------------------------------------------

# feature selection date handle
dateOfOrderDateMonth = pd.DatetimeIndex(X_train["Order Date"]).month
X_train["Order Date"] = dateOfOrderDateMonth
dateOfOrderDateMonth = pd.DatetimeIndex(X_test["Order Date"]).month
X_test["Order Date"] = dateOfOrderDateMonth

dateOfShipDateMonth = pd.DatetimeIndex(X_train["Ship Date"]).month
X_train["Ship Date"] = dateOfShipDateMonth
dateOfShipDateMonth = pd.DatetimeIndex(X_test["Ship Date"]).month
X_test["Ship Date"] = dateOfShipDateMonth

# ------------------------------------------------------------------------------------
# apply encoding on categorical data
ShipMode = 'Ship Mode'
CustomerName = 'Customer Name'
Segment = 'Segment'
Country = 'Country'
City = 'City'
State = 'State'
Region = 'Region'
ProductName = 'Product Name'
OrderID = 'Order ID'
CustomerID = 'Customer ID'
ProductID = 'Product ID'

L1 = MyLabelEncoder(-1)
L2 = MyLabelEncoder(-1)
L3 = MyLabelEncoder(-1)
L4 = MyLabelEncoder(-1)
L5 = MyLabelEncoder(-1)
L6 = MyLabelEncoder(-1)
L7 = MyLabelEncoder(-1)
L8 = MyLabelEncoder(-1)
L9 = MyLabelEncoder(-1)
L10 = MyLabelEncoder(-1)
L11 = MyLabelEncoder(-1)
L12 = MyLabelEncoder(-1)
L13 = MyLabelEncoder(-1)

X_train['MainCategory'] = X_train['CategoryTree'].apply(lambda X_train: eval(X_train)['MainCategory'])
X_train['SubCategory'] = X_train['CategoryTree'].apply(lambda X_train: eval(X_train)['SubCategory'])
X_test['MainCategory'] = X_test['CategoryTree'].apply(lambda X_test: eval(X_test)['MainCategory'])
X_test['SubCategory'] = X_test['CategoryTree'].apply(lambda X_test: eval(X_test)['SubCategory'])
X_train.drop('CategoryTree', axis=1, inplace=True)
X_test.drop('CategoryTree', axis=1, inplace=True)

# drop the original "CategoryTree" column
X_test_columns = X_test.iloc[:, 0:21]

MainCategory = 'MainCategory'
X_train[MainCategory] = Feature_Encoder_train(X_train, MainCategory, L9)
X_test[MainCategory] = Feature_Encoder_test(X_test_columns, MainCategory, L9)

SubCategory = 'SubCategory'
X_train[SubCategory] = Feature_Encoder_train(X_train, SubCategory, L10)
X_test[SubCategory] = Feature_Encoder_test(X_test_columns, SubCategory, L10)

Feature_Encoder_train(X_train, ShipMode, L1)
Feature_Encoder_train(X_train, CustomerName, L2)
Feature_Encoder_train(X_train, Segment, L3)
Feature_Encoder_train(X_train, Country, L4)
Feature_Encoder_train(X_train, City, L5)
Feature_Encoder_train(X_train, State, L6)
Feature_Encoder_train(X_train, Region, L7)
Feature_Encoder_train(X_train, ProductName, L8)
Feature_Encoder_train(X_train, OrderID, L11)
Feature_Encoder_train(X_train, CustomerID, L12)
Feature_Encoder_train(X_train, ProductID, L13)

X_test[ShipMode] = Feature_Encoder_test(X_test, ShipMode, L1)
X_test[CustomerName] = Feature_Encoder_test(X_test, CustomerName, L2)
X_test[Segment] = Feature_Encoder_test(X_test, Segment, L3)
X_test[Country] = Feature_Encoder_test(X_test, Country, L4)
X_test[City] = Feature_Encoder_test(X_test, City, L5)
X_test[State] = Feature_Encoder_test(X_test, State, L6)
X_test[Region] = Feature_Encoder_test(X_test, Region, L7)
X_test[ProductName] = Feature_Encoder_test(X_test, ProductName, L8)
X_test[OrderID] = Feature_Encoder_test(X_test, OrderID, L11)
X_test[CustomerID] = Feature_Encoder_test(X_test, CustomerID, L12)
X_test[ProductID] = Feature_Encoder_test(X_test, ProductID, L13)
# ------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# aplly scaling in the data
# create a StandardScaler object
scaler = StandardScaler()

# fit the scaler to the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)
# ---------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Apply outlier detection on training data
lof = LocalOutlierFactor(n_neighbors=35, contamination=0.1)
y_pred = lof.fit_predict(X_train)
X_train = X_train[y_pred != -1]
y_train = y_train[y_pred != -1]

# Apply outlier detection on testing data
y_pred = lof.fit_predict(X_test)
X_test = X_test[y_pred != -1]
y_test = y_test[y_pred != -1]
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# apply the Correlation
data = pd.concat([X_train, y_train], axis=1)
corr = data.corr()
topFeatures = corr.index[abs(corr['Profit'] > 0.001)]
topcorr = data[topFeatures].corr()
# print(topcorr)
sns.heatmap(topcorr, annot=True)
plt.show()
topFeatures = topFeatures.delete(-1)
X_train = X_train[topFeatures]
X_test = X_test[topFeatures]
# ---------------------------------------------------------------------------------------------


# apply the linearRegression -----------------------------------------------------------------------
linearRegression = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=1)
linearRegression.fit(X_train, y_train)

y_pred = linearRegression.predict(X_test)
y_pred_train = linearRegression.predict(X_train)

# calculate MSE
MSEValue = mean_squared_error(y_test, y_pred, multioutput="uniform_average")
MSEValueTrain = mean_squared_error(y_train, y_pred_train)

print("------------------------------------------------------------------------------------------- ")
print("mean squared error for linear regression model test : ", MSEValue)

print("r2_score for linear regression model: ")
print(r2_score(y_test, y_pred) * 100)
print("------------------------------------------------------------------------------------------- ")
# # scatter plot of actual values
# for col in X_test.columns:
#     plt.scatter(X_test[col], y_test, color="blue")
#
#     # set the title and labels
#     plt.title("Linear Regression Model")
#     plt.xlabel("cols")
#     plt.ylabel("Y")
#
# # plot the regression line
# plt.plot(X_test, y_pred, color="red", linewidth=3)
#
# # show the plot
# plt.show()

# -------------------------------------------------------

# apply polynomialRegression -----------------------------------------------------------------------
poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
y_pred = poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

# calculate MSE
print("------------------------------------------------------------------------------------------- ")
print('Mean Square Error for polynomial regression model', mean_squared_error(y_test, prediction))
print("r2_score polynomial regression model: ")
print(r2_score(y_test, y_pred) * 100)
print("------------------------------------------------------------------------------------------- ")

# # scatter plot of actual values
# for col in X_test.columns:
#     plt.scatter(X_test[col], y_test, color="blue")
#
#     # set the title and labels
#     plt.title("polynomial Regression Model")
#     plt.xlabel("cols")
#     plt.ylabel("Y")
#
# # plot the regression line
# plt.plot(X_test, y_pred, color="red", linewidth=3)
#
# # show the plot
# plt.show()

# -----------------------------------------------------------------------------------------------------------
# Create Ridge Regression model
alpha = 0.1
ridge = Ridge(alpha=alpha)

# Fit the model to the training data
ridge.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ridge.predict(X_test)

# Calculate the Mean Squared Error (MSE) of the predictions
mse = mean_squared_error(y_test, y_pred)
print("------------------------------------------------------------------------------------------- ")
print("Mean Squared Error:", mse)

# Calculate the R-squared (R2) score of the model
r2_score = ridge.score(X_test, y_test)
print("R-squared Score for ridge regression:", r2_score * 100)
print("------------------------------------------------------------------------------------------- ")
# # scatter plot of actual values
# for col in X_test.columns:
#     plt.scatter(X_test[col], y_test, color="blue")
#
#     # set the title and labels
#     plt.title("ridge Regression Model")
#     plt.xlabel("cols")
#     plt.ylabel("Y")
#
# # plot the regression line
# plt.plot(X_test, y_pred, color="red", linewidth=3)
#
# # show the plot
# plt.show()

# ---------------------------------------------------------------------------------------------------------

# Create and fit the Lasso Regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = lasso.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("------------------------------------------------------------------------------------------- ")
print("Mean squared error: {:.2f}".format(mse))

# Get the coefficients of the model
coefficients = lasso.coef_
# print("Coefficients: {}".format(coefficients))
print("R-squared Score for lasso regression:", r2_score * 100)
print("------------------------------------------------------------------------------------------- ")

# # scatter plot of actual values
# for col in X_test.columns:
#     plt.scatter(X_test[col], y_test, color="blue")
#
#     # set the title and labels
#     plt.title("lasso Regression Model")
#     plt.xlabel("cols")
#     plt.ylabel("Y")
#
# # plot the regression line
# plt.plot(X_test, y_pred, color="red", linewidth=3)
#
# # show the plot
# plt.show()

# ---------------------------------------------------------------------

