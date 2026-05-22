from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

# Generate Saliency Maps
def get_saliency_map(model, input_image):
    model.eval()
    input_image.requires_grad_()
    output = model(input_image)
    output = output.sum()
    output.backward()
    saliency_map = input_image.grad.data.abs().max(dim=1)[0]
    return saliency_map


def test_model(y_test, y_pred, save_pred=False, pred_name=None, save_plot=False):
    """
    Evaluates a regression model on test data.
    1. Scatter plot of predictions vs true values
    2. Regression metrics: MSE, MAE, R2
    """

    if save_pred:
        if not os.path.exists('output/predicted_values'):
            os.makedirs('output/predicted_values', exist_ok=True)
        results_df = pd.DataFrame({
            'y_test': np.array(y_test).flatten(),
            'y_pred': np.array(y_pred).flatten()
        })
        results_df.to_csv(f'output/predicted_values/y_{pred_name}.csv', index=False)
        print('saved')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal fit')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Regression Predictions vs True Values')
    ax.legend()

    if save_plot:
        if not os.path.exists('output/figures'):
            os.makedirs('output/figures', exist_ok=True)
        plot_filename = f'output/figures/{pred_name or "test"}.png'
        fig.savefig(plot_filename, bbox_inches='tight')
        print(f'saved plot to {plot_filename}')



    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {rmse}')
    print(f'R2 Score: {r2}')

    return mae, rmse, r2


def test(model, test_dataloader, saliency=False, device='cpu', save=False, save_pred=False, pred_name=None, save_plot=False, target_mean=0.0, target_std=1.0):
    model.to(device)
    model.eval()

    eval_images = []

    with torch.no_grad():
        y_true, y_pred = [], []
        for test_img, test_labels in tqdm(test_dataloader, total=len(test_dataloader)):
            image, labels = test_img.to(device), test_labels.to(device)
            preds = model(image)

            denormlabels = labels.cpu().numpy() * target_std + target_mean
            denormpreds = preds.cpu().numpy() * target_std + target_mean
            y_true.extend(denormlabels)
            y_pred.extend(denormpreds)

            if len(eval_images) < 5:
                for img in image:
                    if len(eval_images) < 5:
                        eval_images.append(img.cpu())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        test_model(y_true, y_pred, save_pred, pred_name, save_plot)

    if saliency:
        if save:
            if not os.path.exists('output'):
                os.makedirs('output', exist_ok=True)
            if not os.path.exists('output/saliency_maps'):
                os.makedirs('output/saliency_maps', exist_ok=True)

        print('#' * 50, f' Saliency Maps ', '#' * 50)
        print('')

        for i, eval_image in enumerate(eval_images):
            eval_image = eval_image.unsqueeze(0)
            saliency_map = get_saliency_map(model, eval_image)

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(eval_image[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.title(f'Original Image {i}')

            plt.subplot(1, 2, 2)
            plt.imshow(saliency_map[0].detach().cpu().numpy(), cmap=plt.cm.hot)
            plt.title('Saliency Map')

            plt.tight_layout()
            if save:
                plt.savefig(f'output/saliency_maps/saliency_map_image_{i}.pdf')
            plt.show()
