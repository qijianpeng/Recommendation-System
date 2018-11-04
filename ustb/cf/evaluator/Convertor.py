#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/11/4.
 Email: jianpengqi@126.com
"""
import numpy as np

class Convertor(object):

  @classmethod
  def transformRatingsToMatrix(cls, R, O):
    """
    转换计算后的用户评分记录与Test data set中获取到的真实评分记录为统一格式.
    :param R: 用户u的计算后的数据
    :param O: dict, Loads from test data. Value like this: user_id:{item_1:item_1_rating, item_2:item_2_rating ... }
    :return: R' and O' transformed from O.
            R' 与 O' shape一致，通过测试集中出现过的item对计算后的结果进行提取而来  
            O' likes this: [item_1_rating, item_2_rating, ...., item_n_rating]
    """
    cols = np.shape(R) # rows should eq to 1
    _O = np.zeros(cols)
    _R = np.zeros(cols)
    for (item_id, item_ratings) in O.items():
      index = item_id - 1 # Remember index starts from 0.
      _O[index] = item_ratings
      _R[index] = R[index] # 我们仅仅需要测试集中出现过的item集
    # Now shrink useless values of R
    return _R[_R>0], _O[_O>0]