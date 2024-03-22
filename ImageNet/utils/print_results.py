import numpy as np
from ood_tool.display_results import get_measures, print_measures, print_measures_with_std

def get_and_print_results(in_score, out_score, num_to_avg=1,auroc_list=None, aupr_list=None, fpr_list=None, method_name=''):
    if auroc_list is None:
        auroc_list = []
    if aupr_list is None:
        aupr_list = []
    if fpr_list is None:
        fpr_list = []
    aurocs, auprs, fprs = [], [], []

    measures = get_measures(-in_score, -out_score)

    aurocs.append(measures[0])
    auprs.append(measures[1])
    fprs.append(measures[2])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs,method_name)
    else:
        print_measures(auroc, aupr, fpr, method_name)
    return auroc_list , aupr_list , fpr_list