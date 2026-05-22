import torch
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def train(model, train_dataloader, val_dataloader, criterion, optimizer, ppath, scheduler=None, num_epochs=50, backbone='Retina', save=False, device='cpu', patience=7, train_sampler=None, is_main_process=True, target_mean=0.0, target_std=1.0):
    # model.to(device)

    train_losses = []
    val_losses = []
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'val_loss': float('inf'),
        'val_mae': float('inf'),
        'val_rmse': float('inf'),
        'val_r2': float('-inf')
    }
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        num_train_batches = len(train_dataloader)

        for train_img, train_labels in tqdm(train_dataloader, total=num_train_batches):
            inputs = train_img.to(device)
            labels = train_labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / num_train_batches
        train_losses.append(avg_train_loss)

        if is_main_process:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for val_img, val_labels in tqdm(val_dataloader, total=len(val_dataloader)):
                    val_inputs = val_img.to(device)
                    val_labels = val_labels.to(device)

                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_labels.float()).item()

                    denorm_preds = val_outputs.cpu().numpy() * target_std + target_mean
                    denorm_targets = val_labels.cpu().numpy() * target_std + target_mean

                    all_preds.extend(denorm_preds)
                    all_targets.extend(denorm_targets)

            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            val_mae = mean_absolute_error(all_targets, all_preds)
            mae_scores.append(val_mae)
            val_rmse = root_mean_squared_error(all_targets, all_preds)
            rmse_scores.append(val_rmse)
            val_r2 = r2_score(all_targets, all_preds)
            r2_scores.append(val_r2)

            if scheduler is not None:
                scheduler.step(val_loss)
        else:
            val_loss = 0.0
            val_mae = 0.0
            val_rmse = 0.0

        if is_main_process:
            print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}, Val MAE: {val_mae}, Val RMSE: {val_rmse}, Val R2: {val_r2}')

            if val_loss < best_model_info['val_loss']:
                best_model_info['epoch'] = epoch + 1
                if train_sampler is not None:
                    best_model_info['state_dict'] = {
                        k: v.cpu() for k, v in model.module.state_dict().items()
                    }
                else:
                    best_model_info['state_dict'] = {
                        k: v.cpu() for k, v in model.state_dict().items()
                    }
                best_model_info['val_loss'] = val_loss
                best_model_info['val_mae'] = val_mae
                best_model_info['val_rmse'] = val_rmse
                best_model_info['val_r2'] = val_r2
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print('Early stopping triggered.')
                early_stop = True
        else:
            early_stop = False

        if train_sampler is not None:
            stop_tensor = torch.tensor(int(early_stop), device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            early_stop = bool(stop_tensor.item())
        if early_stop:
            break

    if is_main_process and not early_stop:
        print('Training completed without early stopping.')

    if best_model_info['state_dict'] is not None:
        if train_sampler is not None:
            model.module.load_state_dict(best_model_info['state_dict'])
        else:
            model.load_state_dict(best_model_info['state_dict'])

    if is_main_process and save:
        os.makedirs(os.path.join(ppath, 'models'), exist_ok=True)
        torch.save(best_model_info['state_dict'], os.path.join(ppath, f'models/{backbone}_best.pth'))

    return model
