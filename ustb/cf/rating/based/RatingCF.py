#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/10/16.
 Email: jianpengqi@126.com
"""
from cf.common.CFBase import CFBase
import numpy as np
from scipy.stats.stats import pearsonr
from scipy import spatial

class RatingCF(CFBase):
  def __init__(self):
    """
    Default do nothing.
    """
  def initEnv(self, user_id, n_results):
    CFBase.initEnv(self, user_id, n_results)
    self.knn_count = int(raw_input("请输入要列入计算的相关用户数量（默认为80）：") or 80)
    self.similar_method = int(raw_input("请输入要采用的相似度计算方式: 1, Cosine. 2, Pearson. 3, Jaccard. (Default 2）：") or 2)

  def compute(self):
    """
    Similarity compute methods:
    1. Cosine
    2. Pearson
    3. Jaccard
    """
    user_id = self.user_id - 1
    Rm = self.Rm
    similar_method = self.similar_method
    user_i_history = Rm[user_id]
    N = len(Rm)
    SM = np.zeros(N)
    # Similarity between ui and other users. 1 * N
    # Pearson correlation coefficients.
    if similar_method == 2:
      for index, user_j in enumerate(Rm):
        SM[index] = pearsonr(user_i_history, user_j)[0]
    elif similar_method == 1:
      for index, user_j in enumerate(Rm):
        SM[index] = 1 - spatial.distance.cosine(user_i_history, user_j) # similarity = 1 - distance
    elif similar_method == 3:
      for index, user_j in enumerate(Rm):
        SM[index] = 1 - spatial.distance.jaccard(user_i_history, user_j)

    self.SM = SM

  def recommend(self):
    user_i_vec = self.SM
    user_id = self.user_id
    Rm = self.Rm
    # find nearest neighbors
    acc_idx = np.argpartition(-user_i_vec, self.knn_count + 1)[:self.knn_count + 1]
    acc_idx = acc_idx[acc_idx != user_id]
    knnNeighborsRating = Rm[acc_idx]
    # Use nearest neighbors info to recommend.
    acc = np.sum((knnNeighborsRating > 0).astype(np.float), axis=0)
    user_rated = (Rm[user_id] > 0)
    acc[user_rated == True] = 0
    return self.convertResult(acc, self.n_results, self.movies), acc






