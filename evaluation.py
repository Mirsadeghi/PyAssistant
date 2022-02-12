import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def accuracy_ml(output, target, thr=.3):
    pred = (torch.nn.Sigmoid()(output)>thr).float()
    y_pred = pred.cpu().numpy()
    y_true = target.cpu().numpy()
    class_metrics = []
    for y_pred_i, y_true_i in zip(y_pred.T, y_true.T):
        class_metrics.append(precision_recall_fscore_support(y_true_i, y_pred_i, average='macro', zero_division=0))
    metrics = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    pred = pred.reshape(1, -1)
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # acc1 = (.5*correct.reshape(-1).float()[target.reshape(-1)==0].mean() + .5*correct.reshape(-1).float()[target.reshape(-1)==1].mean())*100.
    acc1 = correct.reshape(-1).float().mean()*100.
    return metrics, np.array(class_metrics)[:,0:3].astype('float'), acc1

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean(), 100*ap