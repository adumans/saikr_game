# refer : http://www.cnblogs.com/jasonfreak/p/5441512.html
from preprocessing.data_parsing import get_data
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

def raw_data():
    """
    This function will output the raw data parsed from training file.

    :return:
    tracexynp: numpy array of training x, y (dim is (3000, 2, length))
    tracexyt: numpy array of corresponding training t
    targetnp: numpy array of target coordinate
    labelnp: numpy array of label

    """
    tracex, tracey, tracet, target, label = get_data()
    return np.array(tracex), np.array(tracey), np.array(tracet), np.array(target), np.array(label)

def plotDis(dataFalse, dataTrue):
    l0 = 'red'
    l1 = 'blue'
    plt.hist(dataFalse, bins=200, normed=True, color=l0, alpha=.5)
    plt.hist(dataTrue, bins=200, normed=True, color=l1, alpha=.5)
    # plt.hist(dataTrue,  bins=30, normed=True, color=l1, alpha=.7)
    # plt.xlim(dataFalse.min() * 1.1, dataFalse.max() * 1.1)
    plt.show()

    print 'in plot'


def minus_data():
    tracex, tracey, tracet, target, label = raw_data()
    # random data sequence for validation_split in model.fit()
    m = tracex.shape[0]
    index = [i for i in range(m)]
    random.shuffle(index)
    tracex_rdm = np.array(tracex)[index]
    tracey_rdm = np.array(tracey)[index]
    tracet_rdm = np.array(tracet)[index]
    target_rdm = target[index]
    label_rdm = label[index]

    x_std = []
    y_std = []
    x_stdFalse = []
    y_stdFalse = []

    k_abs_mean = []
    k_sub_mean = []
    k_abs_meanFalse = []
    k_sub_meanFalse = []

    t_step_mean = []
    # d_step_mean = []
    # d_step_std = []
    t_step_meanFalse = []
    # d_step_meanFalse = []
    # d_step_stdFalse = []
    i = 0
    for eachx, eachy, eachtarget, eacht, eachL in zip(tracex_rdm, tracey_rdm, target_rdm, tracet_rdm, label_rdm):
        delta_x = (eachx[1:]-eachx[:-1])
        delta_y = (eachy[1:]-eachy[:-1])

        k = delta_y/delta_x
        k_abs = abs(k)
        k_abs_inf = np.isinf(k_abs)
        k_abs_inf_count = np.sum(k_abs_inf)
        k_abs = np.array([x for x in k_abs if str(x) != 'nan' and str(x) != 'inf'])
        # print i, k_abs.mean(), k_abs_inf_count, eachL
        k_sub = k[1:]-k[:-1]
        k_sub = np.array([x for x in k_sub if str(x) != 'nan' and str(x) != 'inf' and str(x) != '-inf'])

        if eachL == 0: # false
            x_stdFalse.append(delta_x.std())
            y_stdFalse.append(delta_y.std())
            k_abs_meanFalse.append(k_abs.mean())
            # k_abs_meanFalse.append(k_abs_inf_count)
            k_sub_meanFalse.append(k_sub.std())
            # t_step_meanFalse.append(np.array(eacht).std())
            t_step_meanFalse.append(eacht[-1])
        else :
            x_std.append(delta_x.std())
            y_std.append(delta_y.std())
            k_abs_mean.append(k_abs.mean())
            # k_abs_mean.append(k_abs_inf_count)
            k_sub_mean.append(k_sub.std())
            # t_step_mean.append(np.array(eacht).std())
            t_step_mean.append(eacht[-1])



    i = i+1
    x_y_std = np.nan_to_num(x_std+y_std)
    x_y_stdFalse = np.nan_to_num(x_stdFalse+y_stdFalse)
    x_stdFalse = np.nan_to_num(x_stdFalse)
    x_std = np.nan_to_num(x_std)
    y_stdFalse = np.nan_to_num(y_stdFalse)
    y_std = np.nan_to_num(y_std)
    k_abs_meanFalse = np.nan_to_num(k_abs_meanFalse)
    k_abs_mean = np.nan_to_num(k_abs_mean)
    k_sub_meanFalse = np.nan_to_num(k_sub_meanFalse)
    k_sub_mean = np.nan_to_num(k_sub_mean)
    t_step_meanFalse = np.nan_to_num(t_step_meanFalse)
    t_step_mean = np.nan_to_num(t_step_mean)
    plotDis(dataFalse=t_step_meanFalse, dataTrue=t_step_mean)

    return label_rdm
    # return tracex_rdm, tracey_rdm, tracet_rdm, target_rdm, label_rdm
    # return np.array(new_x), np.array(new_y), tracet, target, label

if __name__ == '__main__':
    minus_data()