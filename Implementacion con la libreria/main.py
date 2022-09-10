import pandas as pd
import sys
import subprocess
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])

# First, I define the column names for the dataset to use
names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

# I read the dataset and store it in a dataframe from pandas
df = pd.read_csv('winequality-red.csv',names=names , sep=';')

# I select alcohol and ph as my independent variables
x = df[['alcohol','pH']]
# The wine quality will be the dependent variable, the one to be predicted
y = df['quality']

# we create aa linear regression model using the sklearn library
r = linear_model.LinearRegression()
r.fit(x, y)

# now we print out model values 
print('Intercept: ', r.intercept_)
print('Coefficients: ', r.coef_)

# now i will use statsmodel library to get a complete description
# of the linear regression.

x = sm.add_constant(x) # adding a constant

# we fit the model
model = sm.OLS(y, x).fit()


# we create some predictions
predictions = model.predict(x) 

print ("comparing prediction and regular values")

error = mean_squared_error(y, predictions)

print ("The predicted values are the following:")

i = 0
for predict in predictions:
  print("pH:",x.pH[i] , "alcohol:", x.alcohol[i] , "Predicted quality: ", predict)
  i+=1


print ("The prediction error is: ", error) 

