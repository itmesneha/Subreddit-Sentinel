from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.linear_model import LogisticRegression

class BaseEmotionClassifier:
    def __init__(self, model):
        self.model = model
        self.tfidf = TfidfVectorizer(
            ngram_range = (1, 2),
            max_features = 3000
        )

    def fit(self, X_train, y_train):
        X_vec = self.tfidf.fit_transform(X_train)
        self.model.fit(X_vec, y_train)

    def predict(self, X_test):
        X_vec = self.tfidf.transform(X_test)
        return self.model.predict(X_vec)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred)
    
class LogisticRegressionClassifier(BaseEmotionClassifier):
    def __init__(self):
        super().__init__(LogisticRegression(
            max_iter = 1000, 
            C = 1, penalty='l1', 
            solver='liblinear'
        ))

def grid_search_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classifiers = {
        "Logistic Regression": (LogisticRegression(max_iter = 1000, solver='liblinear'), {
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l1', 'l2']
        })
    }

    tfidf_params = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__max_features': [3000, 5000, 8000, 10000],
    }

    for name, (model, clf_params) in classifiers.items():
        print(f"\nTuning {name}...")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', model)
        ])

        # Merge TF-IDF and model params
        param_grid = {**tfidf_params, **clf_params}

        grid = GridSearchCV(
            pipeline,
            param_grid = param_grid,
            scoring = 'f1_defualt',
            cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42),
            n_jobs = -1,
            verbose = 1
        )

        grid.fit(X, y_encoded)

        print(f"Best F1 Score for {name}: {grid.best_score_:.4f}")
        print(f"Best Params: {grid.best_params_}")