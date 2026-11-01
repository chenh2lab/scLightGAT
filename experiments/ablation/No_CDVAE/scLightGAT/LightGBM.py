"""LightGBM stage used by scLightGAT."""

import logging
from copy import deepcopy

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .hparam import (
    LIGHTGBM_BAGGING_FRACTION,
    LIGHTGBM_BAGGING_FREQ,
    LIGHTGBM_BOOSTING_TYPE,
    LIGHTGBM_CLASS_WEIGHT,
    LIGHTGBM_FEATURE_FRACTION,
    LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_MAX_DEPTH,
    LIGHTGBM_METRIC,
    LIGHTGBM_MIN_DATA_IN_LEAF,
    LIGHTGBM_N_ESTIMATORS,
    LIGHTGBM_N_JOBS,
    LIGHTGBM_NUM_LEAVES,
    LIGHTGBM_OBJECTIVE,
    LIGHTGBM_OOF_FOLDS,
    LIGHTGBM_REG_ALPHA,
    LIGHTGBM_REG_LAMBDA,
)


logger = logging.getLogger(__name__)


class LightGBMModel:
    """Train LightGBM with OOF probabilities for downstream GAT fitting."""

    def __init__(self, config=None, random_state=0):
        selected = {
            "bagging_fraction": LIGHTGBM_BAGGING_FRACTION,
            "bagging_freq": LIGHTGBM_BAGGING_FREQ,
            "class_weight": LIGHTGBM_CLASS_WEIGHT,
            "feature_fraction": LIGHTGBM_FEATURE_FRACTION,
            "learning_rate": LIGHTGBM_LEARNING_RATE,
            "max_depth": LIGHTGBM_MAX_DEPTH,
            "min_data_in_leaf": LIGHTGBM_MIN_DATA_IN_LEAF,
            "n_estimators": LIGHTGBM_N_ESTIMATORS,
            "n_jobs": LIGHTGBM_N_JOBS,
            "num_leaves": LIGHTGBM_NUM_LEAVES,
            "reg_alpha": LIGHTGBM_REG_ALPHA,
            "reg_lambda": LIGHTGBM_REG_LAMBDA,
        }
        if config:
            selected.update(deepcopy(config))
        self.class_weight = selected.pop("class_weight")
        self.config = selected
        self.random_state = int(random_state)
        self.model = None
        self.n_classes = None

    def _new_classifier(self, random_state):
        return LGBMClassifier(
            objective=LIGHTGBM_OBJECTIVE,
            boosting_type=LIGHTGBM_BOOSTING_TYPE,
            num_class=self.n_classes,
            metric=LIGHTGBM_METRIC,
            verbose=-1,
            class_weight=self.class_weight,
            random_state=int(random_state),
            deterministic=True,
            force_col_wise=True,
            **self.config,
        )

    def fit(self, features, labels):
        """Generate OOF probabilities, then fit the final full-reference model."""
        labels = np.asarray(labels)
        classes, class_counts = np.unique(labels, return_counts=True)
        self.n_classes = len(classes)
        n_splits = min(int(LIGHTGBM_OOF_FOLDS), int(class_counts.min()))
        if n_splits < 2:
            raise ValueError(
                "OOF LightGBM probabilities require at least two cells per class"
            )

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        class_positions = {label: index for index, label in enumerate(classes)}
        probabilities = np.zeros(
            (len(labels), self.n_classes), dtype=np.float32
        )
        for fold, (train_index, valid_index) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            train_features = (
                features.iloc[train_index]
                if hasattr(features, "iloc")
                else features[train_index]
            )
            valid_features = (
                features.iloc[valid_index]
                if hasattr(features, "iloc")
                else features[valid_index]
            )
            fold_model = self._new_classifier(self.random_state + fold + 1)
            fold_model.fit(train_features, labels[train_index])
            fold_probabilities = fold_model.predict_proba(valid_features)
            for column, label in enumerate(fold_model.classes_):
                probabilities[
                    valid_index, class_positions[label]
                ] = fold_probabilities[:, column]

        oof_predictions = classes[probabilities.argmax(axis=1)]
        oof_accuracy = accuracy_score(labels, oof_predictions)
        logger.info(
            "LightGBM %d-fold OOF accuracy: %.4f", n_splits, oof_accuracy
        )
        self.model = self._new_classifier(self.random_state)
        self.model.fit(features, labels)
        return probabilities

    def predict(self, features):
        self._require_fitted()
        return self.model.predict(features)

    def predict_proba(self, features):
        self._require_fitted()
        return self.model.predict_proba(features)

    def save(self, path):
        self._require_fitted()
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.n_classes = self.model.n_classes_
        return self

    def _require_fitted(self):
        if self.model is None:
            raise ValueError("LightGBM model has not been fitted")
