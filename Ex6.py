import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,
confusion_matrix
data = pd.read_csv("purchase.csv")
x = data.drop("Purchase", axis=1) # Features
y = data["Purchase"] # Target
xtr,xte,ytr,yte = train_test_split(x, y, test_size=0.3, random_state=42)
# Gaussian Naive Bayes (since features are continuous)
model = GaussianNB()
model.fit(xtr,ytr)
# Predictions
ypr = model.predict(xte)
print("Accuracy:", accuracy_score(yte, ypr))
print("\nConfusion Matrix:\n", confusion_matrix(yte, ypr))
print("\nClassification Report:\n", classification_report(yte, ypr))
new = np.array([[45, 48000, 17, 9]])
pred = model.predict(new)
print("Will the customer purchase?", pred[0])
import matplotlib.pyplot as plt
import seaborn as sns
# Set style
sns.set(style="whitegrid")
# Plot BrowsingTime distribution
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(data=data, x="BrowsingTime", hue="Purchase", bins=6,
kde=True, palette="Set1")
plt.title("Browsing Time Distribution by Purchase")
# Plot Income distribution
plt.subplot(1,2,2)
sns.histplot(data=data, x="Income", hue="Purchase", bins=6, kde=True,
palette="Set2")
plt.title("Income Distribution by Purchase")
plt.tight_layout()
plt.show()
