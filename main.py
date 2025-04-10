import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('data.csv')
print(data.isnull())
print(data.isnull().sum())

sns.scatterplot(x='total_bill', y ='tip', data = data)
plt.xlabel('Independent Variables')
plt.ylabel('Dependent Variable')
plt.title('Scatter Plot of Dependent Vs Independent Variables')
plt.show()

x= data[['total_bill']]
y=data['tip']
X_train , X_test, Y_train , Y_test = train_test_split(x,y,test_size=0.2 ,random_state=49)

model = LinearRegression()
model.fit(X_train,Y_train)
y_predict = model.predict(X_test)
meanSq = mean_squared_error(Y_test,y_predict)
r2Score = r2_score(Y_test,y_predict)
print(f"Mean Squre Error: {meanSq}")
print(f"R2 Score: {r2Score}")

sns.scatterplot(x='total_bill', y='tip', data= data)
plt.plot(X_test,y_predict,color='red',linewidth=2)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.title('Linear Regression Line for Total Bill & Tip')
plt.show()

# Currently the values are:
# Mean Squre Error: 1.0699452926174289
# R2 Score: 0.41611960056843744

# For Better Performance
poly= PolynomialFeatures(degree=2)
poly_X = poly.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(poly_X, y, test_size=0.2, random_state=42)
model.fit(X_train, Y_train)
y_predPoly = model.predict(X_test)
meanSq_poly = mean_squared_error(y_test, y_predPoly)
r2Score_poly = r2_score(y_test, y_predPoly)

print(f'Mean Squared Error (Polynomial): {meanSq_poly}')
print(f'R-squared (Polynomial): {r2Score_poly}')
