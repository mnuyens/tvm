import tvm
import numpy as np


def recall(topk=1, num_class=1000):
    """create a recall@topk metric

    Parameters
    ----------
    topk : int
        The evaluation metric

    Returns
    -------
    metric : function rt, dataset
        The result metric
    """
    def _measure(rt, dataset):
        sum_hit = 0
        sum_ind = 0
        for x, y in dataset:
            rt.set_input(0, x)
            rt.run()
            out_shape = (x.shape[0], num_class)
            ypred = rt.get_output(
                0, tvm.nd.empty(out_shape, ctx=tvm.cpu())).asnumpy()
            sum_ind += len(y)
            pred_labels = np.argsort(ypred, axis=1)
            for j in range(min(ypred.shape[1], topk)):
                sum_hit += (
                    pred_labels[:, ypred.shape[1] - 1 -j].flat == y.flat).sum()
        return float(sum_hit) / sum_ind
    return _measure
