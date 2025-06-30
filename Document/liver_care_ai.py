# liver_care_ai.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load dataset
df = pd.read_csv('indian_liver_patient.csv')

# Step 2: Preprocess data
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df = df.dropna()

X = df.drop(['Dataset'], axis=1)
y = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)  # 1 = liver patient, 0 = no liver disease

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Step 6: Save model
joblib.dump(model, 'liver_model.pkl')
