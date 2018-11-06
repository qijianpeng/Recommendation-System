#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/11/4.
 Email: jianpengqi@126.com
 评估算法好坏最经典的指标：
 - 精度(Precision)：
   Precision = |R intersect U| / |U|. R 为用户实际喜欢的物品，U为推荐出的用户喜欢的物品。
 - 召回率(Recall)：
   Recall = |R intersect U| / |R|
 - F1-Measure

 以上是针对二值的，即是与否的问题，涉及到打分的推荐需要考虑采用误差：
 - 均方根误差(RMSE):
  WIKI: https://en.wikipedia.org/wiki/Root-mean-square_deviation
 - 平分绝对误差(MAE)
  与RMSE差别不大，只是范数变为了1。
 - Normalized Lpnorm
  RMSE，MAE是此形式的特例，这里我们采用此种通用的方法进行实现


 接口设计：
   - Evaluator传入的参数应该是计算后的数据与真实数据
   - 数据处理(ETL)在此处不应进行
"""
