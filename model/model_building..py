import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 2. Select specific features for the demo
selected_features = [
    'mean radius', 
    'mean texture', 
    'mean perimeter', 
    'mean area', 
    'mean smoothness', 
    'mean compactness'
]

X = df[selected_features]
y = df['target']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the model to a file named 'breast_cancer_model.pkl'
output_file = 'breast_cancer_model.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Success: Model trained and saved as '{output_file}'")
