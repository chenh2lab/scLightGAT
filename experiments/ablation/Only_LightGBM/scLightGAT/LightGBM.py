"""LightGBM stage used by scLightGAT."""

from copy import deepcopy

import joblib
import numpy as np
from lightgbm import LGBMClassifier

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
    LIGHTGBM_REG_ALPHA,
    LIGHTGBM_REG_LAMBDA,
)


class LightGBMModel:
    """Train the LightGBM classifier used by this ablation."""

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
        """Fit the final classifier on all reference cells."""
        labels = np.asarray(labels)
        self.n_classes = len(np.unique(labels))
        self.model = self._new_classifier(self.random_state)
        self.model.fit(features, labels)
        return self

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
