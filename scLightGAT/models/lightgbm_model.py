#lightgbm_model.py

import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)

class LightGBMModel:
    def __init__(self, use_default_params=True):
        self.model = None
        self.best_params = None
        self.n_class = None
        self.use_default_params = use_default_params

    def optimize_params(self, X, y, n_trials=12):
        self.n_class = len(np.unique(y))
        if self.use_default_params:
            print("Using default parameters")
            self.best_params = {
                'max_depth': 10,
                'num_leaves': 77,
                'learning_rate': 0.2106,
                'n_estimators': 150,
                'min_data_in_leaf': 64,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        else:
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.22),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                }
                
                clf = LGBMClassifier(
                    objective='multiclass',
                    boosting_type='gbdt',
                    metric='multi_logloss',
                    verbose=-1,
                    num_class=self.n_class,
                    **params
                )
                scores = cross_val_score(clf, X, y, cv=5, scoring='neg_log_loss')
                return np.mean(scores)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            self.best_params = study.best_params

        print("Best parameters:", self.best_params)

    def train(self, X, y, test_size=0.3, random_state=42, group_name=None, class_names=None):
        if self.best_params is None:
            self.optimize_params(X, y)

        if self.n_class is None:
            self.n_class = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = LGBMClassifier(
            objective='multiclass',
            boosting_type="gbdt",
            num_class=self.n_class,
            metric="multi_logloss",
            verbose=-1,
            **self.best_params
        )
        self.model.fit(X_train, y_train)

        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        logger.info(f'Test Accuracy: {test_accuracy:.4f}')

        report = classification_report(y_test, y_test_pred)
        logger.info('Classification report:\n' + report)

        plt.figure(figsize=(12, 10))
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        labels = class_names if class_names is not None else np.unique(y)
        sns.heatmap(conf_matrix, 
                    annot=True,
                    fmt='g',
                    cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, ha='right', fontsize=11)
        plt.xlabel('Prediction', fontsize=12, fontweight="bold")
        plt.ylabel('Ground Truth', fontsize=12, fontweight="bold")
        if group_name:
            title = f'"{group_name}" Confusion Matrix'
            filename = f'confusion_matrix_{group_name.replace(" ", "_")}.png'
        else:
            title = 'LightGBM Confusion Matrix'
            filename = 'lightgbm_confusion_matrix.png'
        plt.title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)

from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_lightgbm(X, y, model_dir):
    model = LightGBMModel(use_default_params=True)
    model.optimize_params(X, y)
    model.train(X, y)

    encoder = LabelEncoder()
    encoder.fit(y)

    joblib.dump(model, os.path.join(model_dir, "lightgbm_model.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "label_encoder.pkl"))

    return model, encoder

def load_lightgbm_model(model_dir):
    model = joblib.load(os.path.join(model_dir, "lightgbm_model.pkl"))
    encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return model, encoder
