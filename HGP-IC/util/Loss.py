import torch


def cross_entropy(pred, true):###一个batch的平均的loss

    def c(y_pred, y_true):
        C = 0
        # one-hot encoding
        for col in range(y_true.shape[-1]):
            y_pred[col] = y_pred[col] if y_pred[col] < 1 else 0.99999
            y_pred[col] = y_pred[col] if y_pred[col] > 0 else 0.00001
            C += y_true[col]*torch.log(y_pred[col])+(1-y_true[col])*torch.log(1-y_pred[col])
        return -C

    for i in range(pred.shape[0]):
        if i == 0:
            loss = c(pred[i], true[i])
        else:
            loss += c(pred[i], true[i])
    return loss/pred.shape[0]
