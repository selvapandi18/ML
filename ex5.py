import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,
confusion_matrix
data = pd.read_csv("heart.csv")
x = data.drop("HeartDisease", axis=1) # target column has heart disease labels
y = data["HeartDisease"]
xtr,xte,ytr,yte = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
xtr = scaler.fit_transform(xtr)
xte = scaler.transform(xte)
import matplotlib.pyplot as plt
# Test accuracy for k = 1 to 3
accuracies = []
for k in range(1, 4):
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(xtr, ytr)
acc = knn.score(xte, yte)
accuracies.append(acc)
plt.figure(figsize=(8,5))
plt.plot(range(1,4), accuracies, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k in k-NN")
plt.grid(True)
plt.show()
ypr = knn.predict(xte)
print(ypr)
# Evaluation
print("Accuracy:", accuracy_score(yte, ypr))
print("\nConfusion Matrix:\n", confusion_matrix(yte, ypr))
print("\nClassification Report:\n", classification_report(yte, ypr))
