import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("credit_risk_dataset.csv")
condition = (df['person_age'] > 100) | (df['person_emp_length'] > 60)
df_1 = df[~condition].copy()

# Encoding for categorical features
# person_home_ownership
df_1 = pd.get_dummies(df_1, columns=['person_home_ownership'], drop_first=False, dtype=int)

# loan_intent
df_1 = pd.get_dummies(df_1, columns=['loan_intent'], drop_first=False, dtype=int)

# loan_grade
grade_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df_1['loan_grade_enc'] = df_1['loan_grade'].map(grade_order)

# cb_person_default_on_file
df_1['cb_person_default_on_file_enc'] = df_1['cb_person_default_on_file'].map({'Y': 1, 'N': 0})


df_1 = df_1.drop(columns = ['loan_grade', 'cb_person_default_on_file'])



# Split the dataset into a training dataset and testing dataset
train_df, test_df = train_test_split(
    df_1, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_1['loan_status']
)
train_df = train_df.reset_index(drop=True)


# features and target
features = list(train_df.columns)
target = 'loan_status'
features.remove(target)


# X_train, y_train
X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

X_train.head().T



# Training model
rf_steps = [
    ('rf', RandomForestClassifier(
        class_weight='balanced',
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=2,
        n_estimators=100,
        random_state=42
    ))
]
rf_model = Pipeline(rf_steps)
rf_model.fit(X_train, y_train)


# Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)