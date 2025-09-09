import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


df = pd.read_csv("ex_4.csv")

print("\nColumns in CSV:", df.columns.tolist())
print("\nFirst 5 rows of data:\n", df.head())


df['Attendance'] = df['Attendance'].str.replace('%', '', regex=False).astype(float)



x = df[['Study_Hr', 'Attendance']]
y = df['Result']


clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(x, y)


plt.figure(figsize=(8, 6))
plot_tree(
    clf,
    feature_names=['Study_Hr', 'Attendance'],
    class_names=clf.classes_,
    filled=True
)
plt.show()

new_data = [[5, 85]]
pred = clf.predict(new_data)
print("\nPrediction for new student:", pred[0])
