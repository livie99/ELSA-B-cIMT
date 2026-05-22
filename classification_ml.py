# %% [markdown]
# ### Setup Environment:
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 6"
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# %%

def load_embeddings(dataset, model_name):

    train_embeddings = torch.load(
        os.path.join(dataset, f'extracted_feature/train_embeddings_{model_name}.pt'),
        weights_only=False,
    )
    val_embeddings = torch.load(
        os.path.join(dataset, f'extracted_feature/val_embeddings_{model_name}.pt'),
        weights_only=False,
    )
    test_embeddings = torch.load(
        os.path.join(dataset, f'extracted_feature/test_embeddings_{model_name}.pt'),
        weights_only=False,
    )

    X_train = np.array(train_embeddings['features'])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = np.array(train_embeddings['labels'])

    X_val = np.array(val_embeddings['features'])
    X_val = scaler.transform(X_val)
    y_val = np.array(val_embeddings['labels'])

    X_test = np.array(test_embeddings['features'])
    X_test = scaler.transform(X_test)
    y_test = np.array(test_embeddings['labels'])

    y_train = [int(x > 0.7) for x in y_train]
    y_val = [int(x > 0.7) for x in y_val]
    y_test = [int(x > 0.7) for x in y_test]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    print(y_test, y_test_pred)

    def get_probabilities(model, X):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return None

    def safe_roc_auc_score(y_true, y_score):
        try:
            return roc_auc_score(y_true, y_score)
        except ValueError:
            return None

    y_val_prob = get_probabilities(model, X_val)
    y_test_prob = get_probabilities(model, X_test)

    metrics = {
        'val_f1': f1_score(y_val, y_val_pred),
        'val_auc': safe_roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else None,
        'val_prc': average_precision_score(y_val, y_val_prob) if y_val_prob is not None else None,
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': safe_roc_auc_score(y_test, y_test_prob) if y_test_prob is not None else None,
        'test_prc': average_precision_score(y_test, y_test_prob) if y_test_prob is not None else None,
    }

    print(
        f"{name} - Validation AUROC: {metrics['val_auc']:.4f}, AUPRC: {metrics['val_prc']:.4f}, F1: {metrics['val_f1']:.4f}, "
    )
    print(
        f"{name} - Test AUROC: {metrics['test_auc']:.4f}, AUPRC: {metrics['test_prc']:.4f}, F1: {metrics['test_f1']:.4f}, "
    )

    return metrics


def run_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = LogisticRegression(C = 0.05, max_iter=5000, random_state=42)
    return train_and_evaluate('Logistic Regression', clf, X_train, y_train, X_val, y_val, X_test, y_test)


def run_svc(X_train, y_train, X_val, y_val, X_test, y_test):
    svc = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    return train_and_evaluate('Support Vector Classifier', svc, X_train, y_train, X_val, y_val, X_test, y_test)


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    return train_and_evaluate('Random Forest Classifier', rf, X_train, y_train, X_val, y_val, X_test, y_test)


def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    xgb = XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )
    return train_and_evaluate('XGBoost Classifier', xgb, X_train, y_train, X_val, y_val, X_test, y_test)

def main():
    dataset = os.getcwd()
    model_name = 'retfound_d2_m'
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings(dataset, model_name)

    run_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    run_svc(X_train, y_train, X_val, y_val, X_test, y_test)
    run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == '__main__':
    main()
