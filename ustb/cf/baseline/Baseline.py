#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/10/20.
 Email: jianpengqi@126.com
"""
from cf.common.CFBase import CFBase
import numpy as np

class Baseline(CFBase):

  def __init__(self):
    """Do nothing"""

  def initEnv(self, user_id, n_results):
    CFBase.initEnv(self, user_id, n_results)
    self.lamada2 = float(raw_input("请输入lamada2(Default is 0)：") or 0)
    self.lamada3 = float(raw_input("请输入lamada3(Default is 0)：") or 0)


  def recommend(self):
    user_i_vec = self.SM
    user_rated = (self.Rm[self.user_id - 1] > 0)
    user_i_vec[user_rated == True] = 0
    res = self.convertResult(user_i_vec, self.n_results, self.movies)
    return res, user_i_vec


  def compute(self):
    Rm = self.Rm # user rating matrix.
    lamada2 = self.lamada2 # default 0
    lamada3 = self.lamada3 # default 0
    # 1. computes μ 计算所有评分的平均值
    avg = np.average(Rm)
    # average errors, R_avg_error[u, i] references to Rate(item_i) for user u.
    # 计算平均差矩阵， 即 Rm_{i,j} = Rm_{i, j} - μ
    R_avg_error = np.where(Rm > 0, Rm - avg, Rm)
    # 2. computes bi
    #2.1 Compute sum of squared errors for each item.
    #  SI[i] references to sum(errors(item_i))
    SI = np.sum(R_avg_error, axis = 0)
    #2.2 Count number of rated for each item.
    #  CI[i] references to count( Rate(item_i) > 0 )
    CI = np.sum((Rm > 0).astype(np.int), axis = 0) + lamada2
    BI = SI / CI
    # 3. computes bu
    # 3.1 Count number of items for each user rated.
    # CU[i] references to number of items user_i rated.
    SU = np.sum(np.where(Rm > 0, R_avg_error - BI, 0), axis = 1)
    CU = np.sum( (Rm > 0).astype(np.int),axis = 1) + lamada3
    BU = SU / CU

    # 4. computes bui
    #BUI = avg + BU[user_id] + BI[item_i]
    SM = avg + BU[self.user_id - 1] + BI #得到的为user_id对应用户对所有item的评分
    self.SM = np.where(SM > 5.0, 5.0, SM) #规格化处理，评分最高不会超过5分



