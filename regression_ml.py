# %% [markdown]
# ### Setup Environment:
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 6"
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# %%

def load_crfs(dataset, crf_features):
    def preprocess_crfs(df):
        df['Sex'] = df['Sex'].replace({1: 0, 2: 1})
        df['Smoking'] = df['Smoking'].apply(lambda v: 1 if v == 2 else 0)
        return df
    train_crfs = pd.read_csv(os.path.join(dataset, 'data/tab_202604/train.csv'))[crf_features]
    train_crfs = preprocess_crfs(train_crfs)

    val_crfs = pd.read_csv(os.path.join(dataset, 'data/tab_202604/valid.csv'))[crf_features]
    val_crfs = preprocess_crfs(val_crfs)

    test_crfs = pd.read_csv(os.path.join(dataset, 'data/tab_202604/test.csv'))[crf_features]
    test_crfs = preprocess_crfs(test_crfs)

    return train_crfs, val_crfs, test_crfs


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

    X_train = train_embeddings['features']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    y_train = np.array(train_embeddings['labels'])
    MEAN = np.mean(y_train)
    STD = np.std(y_train)
    y_train = (y_train - MEAN) / STD

    X_val = val_embeddings['features']
    X_val = scaler.transform(X_val)
    y_val = np.array(val_embeddings['labels'])
    y_val = (y_val - MEAN) / STD

    X_test = test_embeddings['features']
    X_test = scaler.transform(X_test)
    y_test = np.array(test_embeddings['labels'])
    y_test = (y_test - MEAN) / STD

    return X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD


def train_and_evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD):
    model.fit(X_train, y_train)
    # y_val = y_val * STD + MEAN  # Denormalize targets
    y_test = y_test * STD + MEAN  # Denormalize targets
    # y_val_pred = model.predict(X_val)
    # y_val_pred = y_val_pred * STD + MEAN  # Denormalize predictions
    y_test_pred = model.predict(X_test)
    y_test_pred = y_test_pred * STD + MEAN  # Denormalize predictions

    metrics = {
        # 'val_rmse': root_mean_squared_error(y_val, y_val_pred),
        # 'val_r2': r2_score(y_val, y_val_pred),
        # 'val_mae': mean_absolute_error(y_val, y_val_pred),
        'test_rmse': root_mean_squared_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        # 'val_mean': y_val_pred.mean(),
        # 'val_std': y_val_pred.std(),
        'test_mean': y_test_pred.mean(),
        'test_std': y_test_pred.std(),
    }

    print(
        # f"{name} - Validation RMSE: {metrics['val_rmse']:.4f}, R2 Score: {metrics['val_r2']:.4f}, MAE: {metrics['val_mae']:.4f}"
        f"{name} - Test RMSE: {metrics['test_rmse']:.4f}, R2 Score: {metrics['test_r2']:.4f}, MAE: {metrics['test_mae']:.4f}"
    )

    return metrics


def run_lasso(X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD):
    lasso = Lasso(alpha=0.001, max_iter=4000, random_state=42)
    return train_and_evaluate('Lasso Regression', lasso, X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD)


def run_svr(X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD):
    svr = SVR(kernel='rbf', 
              C=10, 
              epsilon=0.05, 
              gamma='scale')
    return train_and_evaluate('SVR', svr, X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD)


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD):
    rf = RandomForestRegressor(n_estimators=100, 
                               max_depth=10, 
                               random_state=42)
    return train_and_evaluate('RFR', rf, X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD)

def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD):
    xgb = XGBRegressor(objective='reg:squarederror', 
                       n_estimators=300, 
                       learning_rate=0.03, 
                       max_depth=4, 
                       subsample=0.8, 
                       colsample_bytree=0.8, 
                       random_state=42)
    
    return train_and_evaluate('XGBoost Regression', xgb, X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD)

def main():
    dataset = os.getcwd()
    model_name = 'dinov3_large'
    FRS_crf_features = ['Age','Sex','Smoking','Diabetes','SBP','Total_cholesterol','HDL','Antihypertensive_med']
    train_crfs, val_crfs, test_crfs = load_crfs(dataset, FRS_crf_features)
    X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD = load_embeddings(dataset, model_name)
    run_lasso(train_crfs, y_train, val_crfs, y_val, test_crfs, y_test, MEAN, STD)
    run_lasso(X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD)
    run_svr(train_crfs, y_train, val_crfs, y_val, test_crfs, y_test, MEAN, STD)
    run_svr(X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD)
    run_random_forest(train_crfs, y_train, val_crfs, y_val, test_crfs, y_test, MEAN, STD)
    run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD)
    run_xgboost(train_crfs, y_train, val_crfs, y_val, test_crfs, y_test, MEAN, STD)
    run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD)

if __name__ == '__main__':
    main()
