#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/10/13.
 Email: jianpengqi@126.com
"""
from pandas import DataFrame
import pandas as pd
from cf.common.CFBase import CFBase
from cf.utils.cfutils import MatrixUtil, DataUtil
import numpy as np

class AccessCF(CFBase):
  """
  按照用户访问记录加权的协同过滤算法个性化电影推荐系统
  """
  def __init__(self):
    """
    do noting
    """
  # def __init__(self, **params):
  #   self.params = params
  #   self.user_id = self.params['user_id'] - 1
  #   self.knn_count = self.params['knn_count'] # input_K
  #   self.n_results = self.params['n_results'] # input_showResult
  #   self.movies = self.params['movies'] # movies matrix
  #   self.Rm = self.params['Rm']
  #
  #   if self.user_id == '':
  #     raise ValueError("Please input user id.")
  #   if self.knn_count == '':
  #     self.knn_count = 80
  #   if self.n_results == '':
  #     self.n_results = 20
  #   if self.Rm.any():
  #     pass
  #   else:
  #     raise ValueError("Movie dict is empty.")

  def initEnv(self, user_id, n_results):
    CFBase.initEnv(self, user_id, n_results)
    self.knn_count = int(raw_input("请输入要列入计算的相关用户数量（默认为80）：") or 80)

  def compute(self):
    # 计算用户user_i与其他用户之间的相似度
    # 相似度的计算为统计用户间共同打过分的item
    # Notes：这里仅仅计算用户i的相似度，得到的结果具体为一行1*M的向量，即SM。。
    #return byAccess_new.findMovie(self.user_id, self.user_dict, self.movie_dict, self.knn_count)
    # SimMatrix = VV'
    #For user_i, we only need rating history of user i.
    user_id = self.user_id - 1 # 输入的id从1开始，矩阵id从0开始
    Rm = self.Rm
    user_i_history = Rm[user_id]
    self.SM = MatrixUtil.dot_and(user_i_history, np.transpose(Rm)) #计算user_i 与其他用户相似度矩阵，返回user_i与每个用户的相似度

  def recommend(self):
    user_similar_vector = self.SM
    user_id = self.user_id - 1
    Rm = self.Rm
    #1. 查找与user_i最相近的top knn_count 个用户。
    # 这里knn_count + 1是因为其中包含了user_i本身，在后面会去掉。
    knnNeighborsIdx = np.argpartition(-user_similar_vector, self.knn_count + 1)[:self.knn_count + 1]
    knnNeighborsIdx = knnNeighborsIdx[knnNeighborsIdx != user_id]
    #2.  利用top knn_count 个用户计算出各个item的分数
    knnNeighborsRating = Rm[knnNeighborsIdx]
    acc = np.sum((knnNeighborsRating > 0).astype(np.int), axis=0)
    # 排除用户阅读过的item
    user_rated_flags = (Rm[user_id] > 0)
    acc[user_rated_flags == True] = 0 # trick的做法，将读过的标记为0，其他未读过的通过之前的步骤全都>0
    #3. 利用计算好item分数的向量，向用户推荐可阅读格式的前n_results个item
    return self.convertResult(acc, self.n_results, self.movies)

# import unittest as ut
# class TestCase(ut.TestCase):
#   if __name__ == '__main__':
#     ut.main()
#
#   def test_accessCf(self):
#     """
#     For test and sample.
#     :return:
#     """
#     users, ratings, movies, full, Rm = DataUtil.loadData()
#     user_id = 1
#     knn_count = 10  # input_K
#     n_results = 10  # input_showResult
#     acf = AccessCF(user_id=user_id, movies=movies, knn_count = knn_count, n_results = n_results, Rm = Rm)
#     acf.compute()
#     res = acf.recommend()
#     acf.showResult(res)
