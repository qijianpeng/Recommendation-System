#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""

 Collaborative-Filtering System terms to find your favorite movies.
 Try: `python driver.py --help` on your command line to see what happens!

 包含两类：User-based与Item-based.

 User-based:
 1, 按照用户访问记录加权的协同过滤算法个性化电影推荐系统. `python driver.py --method=1`
 2, 按照用户评价加权的协同过滤算法个性化电影推荐系统. `python driver.py --method=2`
      相似度计算可选方式：1, Cosine. 2, Pearson. 3, Jaccard
 3, 基于NMF非负矩阵分解的推荐算法个性化电影推荐系统. `python driver.py --method=3`
 4, 基于Baseline的推荐算法个性化电影推荐系统. `python driver.py --method=4`

 Item-based:（适合用户多，Item少的情况。慎选，当前测试集集包含Item较多，计算复杂度为M^3，本Demo约为3分钟）
 5. 基于Item评分的协同过滤算法个性化电影推荐系统. `python driver.py --method=5`
      相似度计算可选方式：1, Cosine. 2, Pearson. 3, Jaccard


 Enjoy!

 Created by qijianpeng on 2018/10/13.
 feedback: jianpengqi@126.com
"""
import argparse
from argparse import RawTextHelpFormatter
import os
import datetime

import numpy as np

from cf.access.based import AccessCF
from cf.baseline import Baseline
from cf.common.CFBase import CFBase
from cf.evaluator.Convertor import Convertor
from cf.evaluator.NormalizedLpnormDeviation import NormalizedLpnormDeviation
from cf.item import ItemBased
from cf.nmf.based import Nmf as UstbNMF
from cf.rating.based import RatingCF
from cf.utils.cfutils import DataUtil


def switch_method(method):
  """
  提供不同算法的Dict。传入整型值，返回对应的算法instance。
  Notes: 对于算法系统的耦合度处理，没有很好的方式，只能是用户来指定具体的算法
    因此，系统面向对算法有一定了解的用户。本系统采用的设计模式可参考：策略模式。
  """
  return {
    1: AccessCF.AccessCF(), #1，按照用户访问记录加权的协同过滤算法个性化电影推荐系统.
    2: RatingCF.RatingCF(), #2, 按照用户评价加权的协同过滤算法个性化电影推荐系统.
    3: UstbNMF.Nmf(), #3, 基于NMF非负矩阵分解的推荐算法个性化电影推荐系统.
    4: Baseline.Baseline(),#4, 基于Baseline的推荐算法个性化电影推荐系统.
    5: ItemBased.ItemBased()#5, 基于Item评分的协同过滤算法个性化电影推荐系统.
  }[method]

if __name__ == "__main__":
  """
  `python driver.py --help` see usages.
  Requirements:
      - Python 2.7
      - Pandas # 数据的ETL
      - numpy # 矩阵计算
      - scipy # 一些相似度使用的公式，如cosine，Pearson等
      - argparse # 传入参数解析
  """

  parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
  parser.add_argument("--method", help="[1|2|3|4|5]\n"+
                                 "1, 按照用户访问记录加权的协同过滤算法个性化电影推荐系统.\n"+
                                 "2, 按照用户评价加权的协同过滤算法个性化电影推荐系统.\n"+
                                 "3, 基于NMF非负矩阵分解的推荐算法个性化电影推荐系统.\n"+
                                 "4, 基于Baseline的推荐算法个性化电影推荐系统.\n"+
                                 "5, 基于Item评分的协同过滤算法个性化电影推荐系统.",
                      type=int)
  parser.add_argument("--data-directory", help="Default is </data> dir")
  args = parser.parse_args()
  #0. 选取要采用的推荐算法
  cf = switch_method(args.method)
  input_userID=int(raw_input("请输入要查询推荐的用户ID：") )
  input_showResult = int(raw_input("请输入要显示的推荐结果数量（默认为20）：") or 20)
  #1. 初始化环境，获取datasets，用户输入的参数等。
  cf.initEnv(user_id = input_userID, n_results = input_showResult)
#  starttime = datetime.datetime.now()
  #2. 计算相似度
  cf.compute()
  #3. 推荐
  res, res_full = cf.recommend()
#  endtime = datetime.datetime.now()
  cf.showResult(res) # print results
  # For evaluating
  evaluator = NormalizedLpnormDeviation()
  #p = 2
  #evaluator.setP(p)
  real = DataUtil.loadUserTestingData()[input_userID]
  R, O = Convertor.transformRatingsToMatrix(res_full, real)
  print "Normalized L", evaluator.p, " norm Deviation is: ", evaluator.evaluate(R, O)

# print 'Time: ', (endtime - starttime).seconds
