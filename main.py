import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data = pd.read_csv('data.csv')
print(data.isnull())
print(data.isnull().sum())

sns.scatterplot(x='total_bill', y ='tip', data = data)
plt.xlabel('Independent Variables')
plt.ylabel('Dependent Variable')
plt.title('Scatter Plot of Dependent Vs Independent Variables')
plt.show()


