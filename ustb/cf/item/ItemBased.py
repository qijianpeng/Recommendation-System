#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/10/20.
 Email: jianpengqi@126.com
"""
from scipy import spatial
from scipy.stats.stats import pearsonr

from cf.common.CFBase import CFBase
import numpy as np

class ItemBased(CFBase):

  def __init__(self):
    """
    Default do nothing.
    """
  def initEnv(self, user_id, n_results):
    CFBase.initEnv(self, user_id, n_results)
    self.similar_method = int(raw_input("请输入要采用的相似度计算方式: 1, Cosine. 2, Pearson. 3, Jaccard. (Default 2）：") or 2)

  def recommend(self):
    IS = self.IS
    user_id = self.user_id - 1
    Rm = self.Rm
    user_i_history = Rm[user_id]
    user_i_rated = user_i_history > 0
    user_i_unrated = np.zeros(len(user_i_history))
    for unrated_item_id, is_rated in enumerate(user_i_rated):
      if is_rated == False:
        # compute `item_id` similarity by user_i_rated
        similarity = np.dot(IS[unrated_item_id, user_i_rated == True], user_i_history[user_i_rated == True])
        user_i_unrated[unrated_item_id] = similarity / np.sum(IS[unrated_item_id, user_i_rated == True])
    return self.convertResult(user_i_unrated, self.n_results, self.movies)

  def compute(self):
    """
    Similarity compute methods:
    1. Cosine
    2. Pearson
    3. Jaccard
    
    Notes: 为了保持程序思路清晰，并未进行优化。
           一个可行的优化方向应为：只针对user_i未评分过的item计算相似度。
    """
    similar_method = self.similar_method
    user_id = self.user_id - 1
    Rm = self.Rm
    users, items = np.shape(Rm)
    IS = np.zeros((items, items))#items similarity, IS[i, j] references to similarity between item_i and item_j.
    Rm_T = np.transpose(Rm)
    if similar_method == 1: # cosine
      for item_i_id, item_i_rates in enumerate(Rm_T):
        for item_j_id, item_j_rates in enumerate(Rm_T):
          IS[item_i_id, item_j_id] = 1 - spatial.distance.cosine(item_i_rates, item_j_rates)
    elif similar_method == 2: # Pearson
      for item_i_id, item_i_rates in enumerate(Rm_T):
        for item_j_id, item_j_rates in enumerate(Rm_T):
          IS[item_i_id, item_j_id] = pearsonr(item_i_rates, item_j_rates)[0]
    elif similar_method == 3: # Jaccard
      for item_i_id, item_i_rates in enumerate(Rm_T):
        for item_j_id, item_j_rates in enumerate(Rm_T):
          IS[item_i_id, item_j_id] = 1 - spatial.distance.jaccard(item_i_rates, item_j_rates)
    self.IS = IS

