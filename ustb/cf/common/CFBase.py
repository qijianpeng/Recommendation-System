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
from common.RS import RS


class CFBase(RS):
  def __init__(self):
    """
    Default do noting
    """
  def __init__(self, ** params):
    self.params = params

  def initEnv(self, user_id, n_results):
    """初始化环境，加载数据集，输入查询用户的id以及想要获取的推荐数目等。
    子类可以重写此方法，重写的同时，需要调用`CFBase.initEnv()`方法，对数据进行加载
    等的操作。
    `users, ratings, movies, full, Rm`已经加载，可通过`self.<val>`调用。

    注：`user_id`此处输入的是真实的用户id，与对应评分矩阵`Rm`应当为`user_id - 1`.
        由于`user_id`涉及到使用`pandas`进行`union`, `projection`等操作，这里不做处理。

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
    # 1. 从指定用户计算后的item评分向量中获取前`n_results`个item的index（已经打过分的item在之前已经剔除）
    # notes：这里使用的类似于快速排序的算法，获取前`n`个未排序的index。(对于所有的item都进行排序的话，复杂度可能过高)
    recommendations_idx_set = np.argpartition(-similarVector, n_results)[:n_results]
    # Sorting recommendations
    # Gets rating items: 'movie_id', 'rating score'
    # 对获取到的前`n`个index与item的id进行封装
    index_rating = np.vstack((recommendations_idx_set + 1, similarVector[recommendations_idx_set])).T
    # 获取最终结果并排序
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
