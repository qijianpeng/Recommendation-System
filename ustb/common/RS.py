#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/11/17.
 Email: jianpengqi@126.com
"""
class RS(object):
  def __init__(self, **params):
    self.params = params

  def initEnv(self):
    """
    init env, you can load data file, input vars, etc.
    :param user_id: 
    :param n_results: 
    :return: 
    """
    # default do nothing.
    pass

  def compute(self):
    """Computing method.
    :return:
    """
    raise NotImplementedError("Compute method not implemented.")

  def recommend(self):
    """
    when you finished computing, you can make some recommendations.
    :return: 
    """
    raise NotImplementedError("Recommend method not implemented.")