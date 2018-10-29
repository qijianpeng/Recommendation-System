#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/10/13.
 Email: jianpengqi@126.com
"""
import pandas as pd
from pandas import DataFrame
import numpy as np
from texttable import Texttable

from cf.utils.cfutils import DataUtil

class CFBase(object):
  def __init__(self):
    """
    Default do noting 
    """
  def __init__(self, ** params):
    self.params = params

  def initEnv(self, user_id, n_results):
    """
    Args:
      `user_id`: Id of user input.
         For `Rm`, `user_id = user_id - 1`.
         For DataFrame(`users`, `ratings`, `movies`, `full`), `user_id`=`user_id`
      
    set up environment and datasets.
    :return: 
    """
    users, ratings, movies, full, Rm = DataUtil.loadData()
    self.users = users
    self.ratings = ratings
    self.movies = movies
    self.full = full
    self.Rm = Rm
    self.user_id = user_id
    self.n_results = n_results

  def compute(self):
    """Computes similarity between user_i and others.
    计算相似度
    For Nmf, its `PM` and `QM`.
    For AccessCF, its a `one-row` * `n-col` vector(SM), `n` is the number of users.
    For RatingCF, same with AccessCF.
    :return: 
    """
    print "Default do nothing."

  def recommend(self):
    """
    推荐item
    :return: `DataFrame('title', 'release_date', 'rating')` contains results of the user wanted.
    """
    print "Default do nothing."

  def convertResult(self, similarVector, n_results, movies ):
    """
    将推荐的产品转换为用户可阅读的格式，按照降序排列
    Args:
    `similarVector`: Possibility vector(1*N, where N is number of movies) for each movie, 
                          computed from nearest neighbors and their records, see method `recommend()`.
    `n_results`: Number of results to return(display).
    `movies`: Movies matrix, it's a DataFrame generated from pandas. see `DataUtil.loadData()`
    """
    recommendations_idx_set = np.argpartition(-similarVector, n_results)[:n_results]
    # Sorting recommendations
    #recommendations_idx_desending = recommendations_idx_set[np.argsort(similarVector[recommendations_idx_set])]
    #recommendations_movie_idx_desending = recommendations_idx_desending + 1
    # Gets rating items: 'movie_id', 'rating score'
    #index_rating = np.vstack((recommendations_movie_idx_desending, similarVector[recommendations_idx_desending])).T
    index_rating = np.vstack((recommendations_idx_set + 1, similarVector[recommendations_idx_set])).T
    res = pd.merge(movies, DataFrame(index_rating, columns=['movie_id', 'rating'])) \
      .sort_values('rating', ascending=False)[['title', 'release_date', 'rating']]
    return res

  def showResult(self, res):
    full = self.full
    user_id = self.user_id
    user_i_rated = full[full["user_id"] == user_id][['title', 'release_date']]
    print("")
    print("所查询的用户曾经访问过的电影为：")
    rows = []
    table = Texttable()  # 创建表格并显示
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t', 'f'])
    table.set_cols_align(["l", "l"])
    rows.append(["电影名称", "发行时间"])
    for row in user_i_rated[user_i_rated['title'] != 'unknown'].fillna(0).values: # Datasets contains "unknown" noises.
      rows.append(row)
    table.add_rows(rows)
    table.set_deco(Texttable.HEADER)
    print(table.draw())

    print("")
    print("个性化推荐系统所推荐的电影为：")
    rows=[]
    table=Texttable()     #创建表格并显示
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t','f','a'])
    table.set_cols_align(["l","l","l"])
    rows.append(["电影名称","发行时间","推荐指数"])
    for row in res.fillna(0).values:
      rows.append(row)
    table.add_rows(rows)
    table.set_deco(Texttable.HEADER)
    print(table.draw())