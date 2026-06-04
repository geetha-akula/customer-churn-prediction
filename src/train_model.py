# Import Libraries

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, precision_score,
recall_score, f1_score, roc_auc_score, precision_recall_curve)


# Paths

BASE_DIR  =  Path(__file__).resolve().parent.parent
DATA_PATH =  BASE_DIR / "data" / "telco_churn.csv"
MODEL_DIR =  BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"


# Loading Data

df = pd.read_csv(DATA_PATH)

# Dropping Columns

drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 
             'Longitude', 'Churn Score', 'Churn Label', 'Churn Reason'] 

df.drop(columns= drop_cols, inplace= True, errors= 'ignore')



X = df.drop('Churn Value', axis = 1)
y = df['Churn Value']


cat_cols = X.select_dtypes(include = ['object', 'category', 'string']).columns
num_cols = X.select_dtypes(include = ['number']).columns


# train-test split

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state= 42, stratify= y)


# Preprocessing

num_pipe  =  Pipeline(steps= [("imputer", SimpleImputer(strategy= "median")),
                    ("scaler", StandardScaler())])
cat_pipe  =  Pipeline(steps= [("imputer", SimpleImputer(strategy= "most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown= "ignore"))])

col_trans  =  ColumnTransformer(transformers = [("num", num_pipe, num_cols),
                                ("cat", cat_pipe, cat_cols)])



# Models


models = {
    "Logistic Regression": LogisticRegression(max_iter= 1000, class_weight= "balanced", random_state= 42),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth= 5, min_samples_split= 10, min_samples_leaf= 5, random_state= 42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators= 200, max_depth= 6, class_weight= "balanced", random_state= 42),
    "XGBoost": XGBClassifier(n_estimators= 300, max_depth= 6, learning_rate= 0.05, subsample= 0.8, colsample_bytree= 0.8, eval_metric= "logloss", random_state= 42),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=31, class_weight='balanced', min_child_samples=20, random_state=42)
}


# Cross Validation

cv  =  StratifiedKFold(n_splits= 5, shuffle= True, random_state= 42)

cv_results = []

param_grids = {
    "Logistic Regression": {
        "model__C": [0.1, 1, 10]
    },
    "LightGBM": {
        "model__num_leaves": [31, 50],
        "model__min_child_samples": [10, 20],
        "model__learning_rate": [0.01, 0.05],
        "model__reg_alpha": [0, 0.1],
        "model__reg_lambda": [0, 0.1],
    },
    "param_grid_xgb" : {
        "model__n_estimators": [200, 300],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8],
        "model__colsample_bytree": [0.7, 0.8],
        "model__scale_pos_weight": [2, 3, 4]
    }
}
    

# Threshold

def best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

# Model Evaluation

def evaluate(model_name, pipeline, X_test, y_test):
    y_prob = pipeline.predict_proba(pd.DataFrame(X_test, columns= X.columns))[:, 1]

    threshold  =  best_threshold(y_test, y_prob)
    y_pred  =  (y_prob >= threshold).astype(int)

    print(f"{model_name}")
    print(f"Best Threshold: {threshold: .4f}")
    print("\n")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision :", round(precision_score(y_test, y_pred), 4))
    print("Recall :", round(recall_score(y_test, y_pred), 4))
    print("F1 score :", round(f1_score(y_test, y_pred), 4))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", round(roc_auc, 4))

    print("\n Classification Report: ")
    print(classification_report(y_test, y_pred))

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return f1_score(y_test, y_pred)

best_models  =    {}
best_model_name  =   None
best_pipeline    =   None
best_f1      =   -1


for model_name, model in models.items():
    pipeline  =  Pipeline(steps= [('preprocessing', col_trans), ('model', model)])

    scores  =  cross_val_score(pipeline, X_train, y_train, cv= cv, scoring= "f1", n_jobs= -1)
    print(f"{model_name}: Mean F1= {scores.mean(): .4f}")

    if model_name in param_grids:
        grid = GridSearchCV(
            pipeline, param_grids[model_name], cv= 3, scoring= "f1", n_jobs= -1
        )
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
        print(f"Best params for {model_name}: {grid.best_params_}")
    else:
        pipeline.fit(X_train, y_train)

    best_models[model_name] = pipeline
    
    current_f1  =  evaluate(model_name, pipeline, X_test, y_test)

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_model_name = model_name
        best_pipeline  =  pipeline


# Voting Classifier

voting_model   =   VotingClassifier(
                        estimators= [
                            ('log_reg', best_models["Logistic Regression"].named_steps['model']),
                            ('xgb', best_models["XGBoost"].named_steps['model']),
                            ('lgbm', best_models["LightGBM"].named_steps['model'])
                        ],
                        voting= 'soft'
)

voting_pipeline = Pipeline([
    ("Preprocessing", col_trans),
    ("model", voting_model)
])

voting_pipeline.fit(X_train, y_train)
voting_f1 = evaluate("Voting Classifier", voting_pipeline, X_test, y_test)

if voting_f1 > best_f1:
    best_f1 = voting_f1
    best_model_name = "Voting Classifier"
    best_pipeline = voting_pipeline


# Training on Whole Dataset
best_pipeline.fit(X, y)


# Saving the Best Model

MODEL_DIR.mkdir(parents= True, exist_ok= True)
joblib.dump(best_pipeline, MODEL_PATH)

print(f"Best model based on f1 score: {best_model_name}")
print(f"Model saved at: {MODEL_PATH}")


