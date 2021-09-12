import numpy as np
import pandas as pd
from pyreadr import pyreadr

from utils.utils import plot_roc_curve, plot_pr

if __name__ == '__main__':
    """
    Script to plot roc curves and pr curves.
    """
    task = 'mortality'

    population = pd.read_pickle(f'./data/{task}/population.pkl')
    outcomes = population['outcomeCount'].values

    indices_test = np.argwhere(population['index'].values == -1.0).squeeze()
    y_test = outcomes[indices_test]

    prediction_dict = dict()
    lasso_predictions_path = f'./results/{task}/lasso_prediction.rds'
    preds = pyreadr.read_r(lasso_predictions_path)[None]
    preds = preds[['rowId', 'value']]
    population = population.join(preds.set_index('rowId'), on='rowId')
    lasso_predictions = population[population['index'] < 0].value.values
    prediction_dict['Lasso'] = lasso_predictions
    prediction_dict['CatBoost'] = np.load(f'./results/{task}/catboost_predictions.npy', allow_pickle=True)
    prediction_dict['RETAIN'] = np.load(f'./results/{task}/RETAIN_predictions.npy')
    prediction_dict['Transformer'] = np.load(f'./results/{task}/transformer_predictions.npy', allow_pickle=True)
    prediction_dict['SARD'] = np.load(f'./results/{task}/SARD_predictions.npy')

    plot_roc_curve(y_test, prediction_dict, title=task)

    plot_pr(y_test, prediction_dict, title=task)
