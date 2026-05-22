# %% [markdown]
# ### Setup Environment:
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 6"
import pandas as pd
import torch
import argparse
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
def get_args():
    parser = argparse.ArgumentParser(description="Set backbone and backbone_mode for the model.")
    parser.add_argument('-i', '--input', default='crfs', type=str, choices=['embeddings', 'crfs', 'combined'], required=False, help="Specify the input of the model.")
    return parser.parse_args()

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

    X_train = train_embeddings['features']
    y_train = torch.tensor(train_embeddings['labels'], dtype=torch.float32)
    MEAN = torch.mean(y_train).item()
    STD = torch.std(y_train).item()
    y_train = (y_train - MEAN) / STD
    X_val = val_embeddings['features']
    y_val = torch.tensor(val_embeddings['labels'], dtype=torch.float32)
    y_val = (y_val - MEAN) / STD
    X_test = test_embeddings['features']
    y_test = torch.tensor(test_embeddings['labels'], dtype=torch.float32)
    y_test = (y_test - MEAN) / STD

    return X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD

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

    return train_crfs.to_numpy(), val_crfs.to_numpy(), test_crfs.to_numpy()
# %%
## MLP Regressor Implementation
class MLPRegressor(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim=1, dropout=0.2):
        super().__init__()
        layers = []
        if isinstance(hidden_dim, int):
            layers.append(torch.nn.Linear(feature_dim, hidden_dim))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            feature_dim = hidden_dim

        elif isinstance(hidden_dim, list):
            for h in hidden_dim:
                layers.append(torch.nn.Linear(feature_dim, h))
                layers.append(torch.nn.ReLU(inplace=True))
                layers.append(torch.nn.Dropout(dropout))
                feature_dim = h

        self.model = torch.nn.Sequential(
            *layers,
            torch.nn.Linear(feature_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

#%%
# 
def train_mlp(X_train, y_train, X_val=None, y_val=None, hidden_dim=[256, 256],
              lr=1e-3, batch_size=32, epochs=300, device=None, save_best=True, save_path='best_mlp.pt',
              MEAN=None, STD=None, early_stopping_patience=20):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = y_train.view(-1, 1)
    # X_train.shape = [n_samples, n_features], y_train.shape = [n_samples, 1]
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MLPRegressor(X_train.shape[1], hidden_dim).to(device)
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=10,
            factor=0.1,
        )

    best_state = model.state_dict()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * features.size(0)

        epoch_loss /= len(train_loader.dataset)

        if X_val is not None and y_val is not None:
            val_loss, metrics = evaluate_mlp(model, X_val, y_val, device=device, MEAN=MEAN, STD=STD)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            scheduler.step(val_loss)

            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} with best val loss {best_val_loss:.4f}.")
                break
        else:
            best_state = model.state_dict()
            scheduler.step(epoch_loss)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}" + (f", Val Loss: {val_loss:.4f}" if X_val is not None else ""))
        if X_val is not None and y_val is not None:
            print(f"Val Metrics: RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
    model.load_state_dict(best_state)
    if save_best and save_path:
        torch.save(best_state, save_path)
    return model


def evaluate_mlp(model, X, y, device=None, MEAN=None, STD=None, save_prob = False, save_path='mlp_predictions.csv'):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = y.view(-1, 1).to(device)

    with torch.no_grad():
        preds = model(X)

    preds = preds.cpu().numpy().reshape(-1)
    y_true = y.cpu().numpy().reshape(-1)
    if save_prob:
        pd.DataFrame({'preds': preds, 'y_true': y_true}).to_csv(save_path, index=False)
    if not MEAN and not STD:
        preds = preds * STD + MEAN
        y_true = y_true * STD + MEAN
    return torch.nn.functional.mse_loss(torch.tensor(preds), torch.tensor(y_true)).item(), {
        'rmse': root_mean_squared_error(y_true, preds),
        'mae': mean_absolute_error(y_true, preds),
        'r2': r2_score(y_true, preds),
    }

# %% 
# test
def main(input="embeddings"):
    BASE_ROOT = os.getcwd()
    print(BASE_ROOT)
    model_name = 'retfound_d2_m'
    X_train, y_train, X_val, y_val, X_test, y_test, MEAN, STD = load_embeddings(BASE_ROOT, model_name)
    if input == "embeddings":
        model = train_mlp(X_train, y_train, X_val, y_val, hidden_dim=[256, 256], lr=1e-3, batch_size=32, epochs=300, device=device, save_best=True, save_path=os.path.join(BASE_ROOT,'models', 'best_mlp.pt'), MEAN=MEAN, STD=STD)
        _, test_metrics = evaluate_mlp(model, X_test, y_test, device=device, MEAN=MEAN, STD=STD, save_prob=True, save_path=os.path.join(BASE_ROOT, 'output', 'predicted_values', f'mlp_test_{input}.csv'))
        

    elif input == "crfs":
        FRS_crf_features = ['Age','Sex','Smoking','Diabetes','SBP','Total_cholesterol','HDL','Antihypertensive_med']
        X_train_crf, X_val_crf, X_test_crf = load_crfs(BASE_ROOT, crf_features=FRS_crf_features)
        model = train_mlp(X_train_crf, y_train, X_val_crf, y_val, hidden_dim=[256, 256], lr=1e-3, batch_size=32, epochs=300, device=device, save_best=True, save_path=os.path.join(BASE_ROOT,'models', 'best_mlp.pt'), MEAN=MEAN, STD=STD)
        _, test_metrics = evaluate_mlp(model, X_test_crf, y_test, device=device, MEAN=MEAN, STD=STD, save_prob=True, save_path=os.path.join(BASE_ROOT, 'output', 'predicted_values', f'mlp_test_{input}.csv'))
    
    print(f"Input: {input}")
    print(f"Test: RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}")
if __name__ == "__main__":
    args = get_args()
    main(input=args.input)