# lightgbm_model.py

import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from scLightGAT.logger_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)

class LightGBMModel:
    """
    LightGBM model wrapper for single-cell RNA sequencing data classification.
    This class supports parameter optimization, training, prediction, and visualization.
    """
    def __init__(self, use_default_params=True):
        """
        Initialize the LightGBM model.
        
        Args:
            use_default_params (bool): Whether to use default parameters or optimize them
        """
        self.model = None
        self.best_params = None
        self.n_class = None
        self.use_default_params = use_default_params

    def optimize_params(self, X, y, n_trials=12):
        """
        Optimize hyperparameters either using defaults or Optuna.
        
        Args:
            X: Feature matrix
            y: Target labels
            n_trials: Number of optimization trials for Optuna
        """
        self.n_class = len(np.unique(y))
        if self.use_default_params:
            logger.info("Using default parameters")
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
            logger.info(f"Optimizing LightGBM parameters with {n_trials} trials")
            
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.22),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 80),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.5)
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
            
        logger.info(f"Best parameters: {self.best_params}")


    def train(self, X, y, test_size=0.3, random_state=42, group_name=None, class_names=None):
        """
        Train the LightGBM model with optimized parameters.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of the dataset to include in the test split
            random_state: Random seed for reproducibility
            group_name: Optional name for the group/cell type being trained
            class_names: Optional list of class names for visualization
                
        Returns:
            float: Test accuracy
        """
        if self.best_params is None:
            self.optimize_params(X, y)

        if self.n_class is None:
            self.n_class = len(np.unique(y))

        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Training LightGBM model with {len(X_train)} samples, {X_train.shape[1]} features")
        
        # Initialize model with optimized parameters
        self.model = LGBMClassifier(
            objective='multiclass',
            boosting_type="gbdt",
            num_class=self.n_class,
            metric="multi_logloss",
            verbose=-1,
            **self.best_params
        )
        
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='multi_logloss'
        )

        # Evaluate model performance
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        logger.info(f'Test accuracy: {test_accuracy:.4f}')

        # Generate detailed classification report
        report = classification_report(y_test, y_test_pred)
        logger.info(f'Classification report:\n{report}')

        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_test_pred, class_names, group_name)
        
        return test_accuracy

    def _plot_confusion_matrix(self, y_true, y_pred, class_names=None, group_name=None):
        """
        Generate and save confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names
            group_name: Optional group name for the title
        """
        plt.figure(figsize=(12, 10))
        conf_matrix = confusion_matrix(y_true, y_pred)
        labels = class_names if class_names is not None else np.unique(y_true)
        
        # Create heatmap
        sns.heatmap(conf_matrix, 
                    annot=True,
                    fmt='g',
                    cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels)
        
        # Customize plot
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, ha='right', fontsize=11)
        plt.xlabel('Prediction', fontsize=12, fontweight="bold")
        plt.ylabel('Ground Truth', fontsize=12, fontweight="bold")
        
        # Set title and filename based on group name
        if group_name:
            title = f'"{group_name}" Confusion Matrix'
            filename = f'confusion_matrix_{group_name.replace(" ", "_")}.png'
        else:
            title = 'LightGBM Confusion Matrix'
            filename = 'lightgbm_confusion_matrix.png'
            
        plt.title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {filename}")

    def predict(self, X):
        """
        Generate predictions for input data.
        
        Args:
            X: Feature matrix to predict
            
        Returns:
            array: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Generate class probabilities for input data.
        
        Args:
            X: Feature matrix to predict
            
        Returns:
            array: Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None):
        """
        Extract feature importance from the trained model.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame: Feature importance sorted by value
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
            
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
        
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = joblib.load(model_path)
        self.n_class = self.model.n_classes_
        logger.info(f"Model loaded from {model_path}")


def train_lightgbm(X, y, model_dir):
    """
    Train a LightGBM model and save it to disk.
    
    Args:
        X: Feature matrix
        y: Target labels
        model_dir: Directory to save the model
        
    Returns:
        tuple: Trained model and label encoder
    """
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize and train the model
    model = LightGBMModel(use_default_params=True)
    model.optimize_params(X, y)
    model.train(X, y)

    # Fit label encoder for consistent class mapping
    encoder = LabelEncoder()
    encoder.fit(y)

    # Save model and encoder
    model_path = os.path.join(model_dir, "lightgbm_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    
    logger.info(f"Model and encoder saved to {model_dir}")
    return model, encoder

def load_lightgbm_model(model_dir):
    """
    Load a trained LightGBM model and label encoder from disk.
    
    Args:
        model_dir: Directory containing the saved model
        
    Returns:
        tuple: Loaded model and label encoder
    """
    model_path = os.path.join(model_dir, "lightgbm_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    
    logger.info(f"Model and encoder loaded from {model_dir}")
    return model, encoder