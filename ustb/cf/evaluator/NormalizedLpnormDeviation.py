#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/11/4.
 Email: jianpengqi@126.com
"""
import inspect

from cf.evaluator.base.Evaluator import Evaluator
import numpy as np

"""
p = 2: RMS(Root Mean Squared Error)
p = 1: MAE(Mean Absolute Error)
"""
class NormalizedLpnormDeviation(Evaluator):

  def __init__(self):
    self.p = 2


  def setP(self, p):
    """
    default p is 2.
    :param p:
    :return:
    """
    self.p = p

  def evaluate(self, computed, origin):
    assert np.shape(computed) == np.shape(origin), "Parameters shape not same."
    size = np.size(computed)
    errors = origin - computed
    value = np.linalg.norm(errors, ord=self.p, axis=None, keepdims=False) / size
    return value
