import numpy as np


def xgb_score(loss, preds, labels):
    """
    According to xgboost scoring function
    score = 0.5* G**2 / (H+l2_regularization)
    """
    G = loss.grad(preds, labels).sum()
    H = loss.hess(preds, labels).sum()
    return 0.5 * (G ** 2) / (H + loss.l2_regularization)


def mse_score(labels):
    labels_mean = np.mean(labels)
    return np.mean(np.square(labels - labels_mean))


def entropy_score(labels):
    """
    entropy = sum(p*log(1/p))
    """
    n_labels = labels.shape[0]
    if n_labels <= 1:
        return 0.0

    counts = np.bincount(labels)
    probs = counts / float(n_labels)
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0.0

    entropy = 0.0
    for p in probs:
        entropy -= p*np.log(p)

    return entropy


def xgb_score_gain(loss, left_preds, left_labels, right_preds, right_labels):
    left_score = xgb_score(loss, left_preds, left_labels)
    right_score = xgb_score(loss, right_preds, right_labels)
    origin_score = xgb_score(loss, np.append(left_preds, right_preds), np.append(left_labels, right_labels))
    return left_score + right_score - origin_score


def mse_score_gain(left_labels, right_labels):
    left_score = mse_score(left_labels)
    right_score = mse_score(right_labels)
    origin_score = mse_score(np.append(left_labels, right_labels))
    return left_score + right_score - origin_score


def entropy_score_gain(left_labels, right_labels):
    left_score = entropy_score(left_labels)
    right_score = entropy_score(right_labels)
    origin_score = entropy_score(np.append(left_labels, right_labels))
    return left_score + right_score - origin_score








