def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())