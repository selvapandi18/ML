import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'D:\EX 3.csv')

# Clean column names (remove spaces and make lowercase)
df.columns = df.columns.str.strip().str.lower()

# Encode gender (Male=1, Female=0)
if 'gender' in df.columns:
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
else:
    raise KeyError("The dataset does not contain a 'gender' column after cleaning.")

# Separate features and target
x = df[['age', 'gender', 'bmi', 'bp', 'cholesterol']]
y = df['condition']

# Scale features
scaler = StandardScaler()
xscale = scaler.fit_transform(x)

# Train-test split
xtr, xte, ytr, yte = train_test_split(xscale, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(xtr, ytr)

# Predictions
ypr = model.predict(xte)
yprob = model.predict_proba(xte)[:, 1]  

# Metrics
print("Accuracy:", accuracy_score(yte, ypr))
print("Classification Report:\n", classification_report(yte, ypr, zero_division=1))

# Confusion matrix
cm = confusion_matrix(yte, ypr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Predict for new data
new = pd.DataFrame([[60, 1, 27, 130, 200]], columns=['age', 'gender', 'bmi', 'bp', 'cholesterol'])
newscale = scaler.transform(new)
newcondition = model.predict_proba(newscale)[0][1]

print(f"Probability of developing the condition: {newcondition:.2f}")


 

 

 

 

 

 
