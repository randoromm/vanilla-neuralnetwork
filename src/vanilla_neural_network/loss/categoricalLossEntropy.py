import numpy as np

from src.vanilla_neural_network.loss.loss import Loss


class CategoricalLossEntropy(Loss):
    def forward(self, y_prediction, y_true):
        samples = len(y_prediction)
        # Clip predictions from number very close to zero to number higher
        y_pred_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            # Vector was passed as y_true: [0, 1, 1]
            correct_predictions = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # Onehot encoding was passed as y_true [[1,0,0], [0,1,0], [0,1,0]]
            correct_predictions = np.sum(y_pred_clipped * y_true, axis=1)
        neg_log_likelihoods = -np.log(correct_predictions)
        return neg_log_likelihoods
