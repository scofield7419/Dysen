import math
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import BoolTensor, FloatTensor


def action_planning_reliability(gt_action_plans, pred_action_plans):
    """
    To assess the coincidence between the ChatGPT-generated action schedules and the gold one.
    :param gt_action_plans: N instances
    :param pred_action_plans: N instances
    """
    score = []
    for gt_action_plan, pred_action_plan in zip(gt_action_plans, pred_action_plans):
        _action_score = 0.0
        for i in range(len(pred_action_plan)):
            pred_s = int(pred_action_plan[i][3].split('v')[0])
            pred_e = int(pred_action_plan[i][3].split('v')[0])
            gt_s = 0
            gt_e = 0
            for j in range(len(gt_action_plan)):
                if pred_action_plan[i][:3] == gt_action_plan[j][:3]:
                    gt_s = int(gt_action_plan[j][3].split('v')[0])
                    gt_e = int(gt_action_plan[j][3].split('v')[0])
                    break
            _action_score += 1 / math.sqrt((math.pow(abs(pred_s-gt_s), 2)+math.pow(abs(pred_e-gt_e), 2)))
        score.append(_action_score)
    return torch.tensor(score)


def triplet_recall(gt_dsgs, pred_dsgs):
    """
    To measure the imagination rationality by comparing the ChatGPT-imagined triplets with the gold one.
    :param gt_dsgs: N instances
    :param pred_dsgs： N instances
    """
    score = []
    for gt_dsg, pred_dsg in zip(gt_dsgs, pred_dsgs):
        g_index = 0
        dsg_score = 0.0
        for i in range(len(pred_dsg)):
            pred_sg = pred_dsg[i]

            def find_gt_sg(p_sg, gt_dsg, start_index):
                correct_r = 0
                index = start_index
                for k in range(start_index, len(gt_dsg)):
                    _tmp_correct_r = 0
                    gt_sg = gt_dsg[k]
                    for r in p_sg:
                        if r in gt_sg:
                            _tmp_correct_r += 1
                    if _tmp_correct_r > correct_r:
                        correct_r = _tmp_correct_r
                        index = k
                return index, correct_r

            idx, pred_tri_num = find_gt_sg(pred_sg, gt_dsg, g_index)
            gt_tri_num = len(gt_dsg[idx])
            # pred_tri_num = 0
            _s = pred_tri_num / gt_tri_num
            dsg_score += _s
            g_index = idx

        score.append(dsg_score)
    return torch.tensor(score)
