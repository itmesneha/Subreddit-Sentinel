import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
SEED = 42

class CustomStackingClassifier(BaseEstimator, ClassifierMixin):    
    def __init__(self, base_models, meta_model, feature_sets=None, cv=5, random_state=SEED):
        self.base_models = base_models
        self.meta_model = meta_model
        self.feature_sets = feature_sets
        self.cv = cv
        self.random_state = random_state
    
    def _validate_feature_sets(self, X):
        if self.feature_sets is None:
            raise ValueError("feature_sets cannot be None. Each base model requires specific features.")
        
        for model_name, _ in self.base_models:
            if model_name not in self.feature_sets:
                raise ValueError(f"No feature set provided for model '{model_name}'")
            
            if not isinstance(self.feature_sets[model_name], list) or len(self.feature_sets[model_name]) != 2:
                raise ValueError(f"Feature set for '{model_name}' should be a list [X_train, X_test]")
            
            X_train_model = self.feature_sets[model_name][0]
            if X_train_model.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Feature set for '{model_name}' has {X_train_model.shape[0]} samples, "
                    f"but expected {X.shape[0]} samples"
                )
    
    def fit(self, X, y):
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=True)
        self._validate_feature_sets(X)
        
        # Store number of classes
        self.n_classes_ = len(np.unique(y))
        
        # Create model instances
        self.base_model_instances_ = {}
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        
        # Create a stratified k-fold cross-validator
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Initialize array for meta-features
        meta_features = np.zeros((n_samples, n_models * self.n_classes_))
        
        # Generate meta-features through cross-validation
        for i, (model_name, model) in enumerate(self.base_models):
            if model is None:
                warnings.warn(f"Model instance for '{model_name}' is None.")
                continue
            
            # Get the appropriate feature set for this model
            X_train_model = self.feature_sets[model_name][0]
            
            # For each fold
            for train_idx, val_idx in kf.split(X, y):
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                if hasattr(X_train_model, 'iloc'):
                    # For pandas DataFrame
                    X_train_fold = X_train_model.iloc[train_idx]
                    X_val_fold = X_train_model.iloc[val_idx]
                else:
                    # For numpy arrays and other array-like objects
                    X_train_fold = X_train_model[train_idx]
                    X_val_fold = X_train_model[val_idx]
                
                # Train the model on the training set of this fold
                model_fold = clone(model)
                model_fold.fit(X_train_fold, y_train_fold)
                
                # Get predictions on the validation set
                val_proba = model_fold.predict_proba(X_val_fold)
                
                # Store predictions as meta-features
                meta_features[val_idx, i*self.n_classes_:(i+1)*self.n_classes_] = val_proba
            
            # Now train the final base model on the entire training set
            self.base_model_instances_[model_name] = clone(model).fit(X_train_model, y)
        
        # Fit the meta-model on the meta-features
        self.meta_model_ = clone(self.meta_model).fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['base_model_instances_', 'meta_model_', 'n_classes_'])
        
        # Validate test data dimensions
        X = check_array(X, accept_sparse=True)
        
        # Generate meta-features for prediction
        meta_features = self._get_meta_features(X)
        
        # Make final prediction using the meta-model
        return self.meta_model_.predict(meta_features)
    
    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['base_model_instances_', 'meta_model_', 'n_classes_'])
        
        # Validate test data dimensions
        X = check_array(X, accept_sparse=True)
        
        # Generate meta-features for prediction
        meta_features = self._get_meta_features(X)
        
        # Make final prediction using the meta-model
        return self.meta_model_.predict_proba(meta_features)
    
    def _get_meta_features(self, X):
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models * self.n_classes_))
        
        for i, (model_name, _) in enumerate(self.base_models):
            if model_name not in self.base_model_instances_:
                continue
                
            # Get the appropriate feature set for this model
            try:
                X_model = self.feature_sets[model_name][1]
                if X_model.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"Test feature set for '{model_name}' has {X_model.shape[0]} samples, "
                        f"but expected {X.shape[0]} samples"
                    )
            except (KeyError, IndexError):
                raise ValueError(f"Invalid feature set for model '{model_name}' during prediction")
            
            # Get model predictions
            model = self.base_model_instances_[model_name]
            meta_features[:, i*self.n_classes_:(i+1)*self.n_classes_] = model.predict_proba(X_model)
        
        return meta_features