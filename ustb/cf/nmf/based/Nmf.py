#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Computes nmf.

"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from cf.common.CFBase import CFBase
from cf.utils.cfutils import DataUtil


class Nmf(CFBase):
  def initEnv(self, user_id, n_results):
    CFBase.initEnv(self, user_id, n_results)
    self.max_iter = int(raw_input("Enter the max iterations(30 as default) : ") or 30)
    self.alpha = float(raw_input("Step-size of gradient descent. Default is 2e-2 : ") or 2e-2)
    self.beta = float(raw_input("Parameter of regularization term. Default is 2e-2: ") or 2e-2)
    self.rank = int(raw_input("Parameter of rank. Default is 30: ") or 30)
    self.loss = float(raw_input("Parameter of regularization term. Default is 2e-2: ") or 2e-2)


  def compute(self):
    """
    `Rm = PM * QM`
    :param Rm: N * M rating matrix to estimate.

    :param rank: `Q = P_{N * rank} x Q_{rank * M}`
    :type rank: `int`

    :param max_iter: Maximum number of factorization iterations. Default is 30.
    :type max_iter: `int`

    :param alpha: Step-size of gradient descent. Default is 2e-2.
    :type alpha: `float`

    :param beta: Parameter of regularization term. Default is 2e-2
    :type beta: `float`

    :param loss: Minimal required improvement of the residuals from the previous iteration. 
                 It will break up iterations when residuals smaller than :param: `loss`.
                 Default is 1e-2.
    :type loss: `float`
    
    :return `PM`, `QM`
    """
    Rm = self.Rm
    rank = self.rank
    max_iter = self.max_iter
    alpha = self.alpha
    beta= self.beta
    loss = self.loss
    # Number of rows and cols
    Uc, Oc = np.shape(Rm)
    # matrix factors
    PM = np.random.rand(Uc, rank)
    QM = np.random.rand(rank, Oc)

    for s in range(0, max_iter):
      # Gradient descent（Euclidean distance)）
      # H = H .* (W'*V) ./ ((W'*W)*H);
      PMT = PM.transpose()
      QM = np.multiply(QM, np.dot(PMT, Rm)) / np.dot(np.dot(PMT, PM), QM)

      # W = W .* (V*H') ./ (W*(H*H'));
      QMT = QM.transpose()
      PM = np.multiply(PM, np.dot(Rm, QMT)) / np.dot(PM, np.dot(QM, QMT))

      # compute losses(SSE)
      PQ = np.dot(PM, QM)
      E = Rm - PQ

      #### Magic codes ???
      # TODO explain these chunks. Converts idea from java.
      PMmul = beta / 2 * np.dot(PM, np.transpose(PM))
      QMmul = beta / 2 * np.dot(QM, np.transpose(QM))
      for i in range(0, Uc):
        r_i = np.sum(PMmul[i, :])
        E[i, :] +=  r_i
      for j in range(0, rank):
        c_j = np.sum(QMmul[:, j])
        E[:, j] += c_j


      if np.sum(E) < loss:
        print "OK"
        break
    self.PM = PM
    self.QM = QM

  def recommend(self):
    user_id = self.user_id - 1
    PM = self.PM
    Rm = self.Rm
    user_i_vec = np.dot(PM[user_id, :], self.QM)
    user_rated = (Rm[user_id] > 0)
    user_i_vec[user_rated == True] = 0
    res = self.convertResult(user_i_vec, self.n_results, self.movies)
    return res

# import unittest as ut
# class TestCase(ut.TestCase):
#   if __name__ == '__main__':
#     ut.main()
#
#   def test_nmf(self):
#     """
#     For test and sample.
#     :return:
#     """
#     #users, ratings, movies, Rm = DataUtil.loadData()
#     user_id = 1
#     #knn_count = 10  # input_K
#     n_results = 10  # input_showResult
#     #acf = Nmf(user_id=user_id, movies=movies, rank=30, max_iter=30, n_results = n_results, Rm = Rm)
#     acf = Nmf()
#     acf.initEnv(user_id, n_results)
#     acf.compute()
#     res = acf.recommend()
#     acf.showResult(res)