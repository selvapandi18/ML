import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns


df = pd.read_csv(r'D:\EX2.csv')
df.columns = df.columns.str.strip()  
print(df.columns)  

x = df[['Bed Rooms', 'Size', 'Age', 'Zip Code']]
y = df['Selling Price']


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Zip Code'])], remainder='passthrough')
xen = ct.fit_transform(x)


xtr, xte, ytr, yte = train_test_split(xen, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(xtr, ytr)


ypr = model.predict(xte)
print(ypr)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


plt.figure(figsize=(8, 6))
sns.scatterplot(x=yte, y=ypr, color='blue', s=100)
plt.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()


sns.heatmap(df[['Bed Rooms', 'Size', 'Age']].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
