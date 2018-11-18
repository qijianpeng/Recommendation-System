#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
"""
 Created by qijianpeng on 2018/11/16.
 Email: jianpengqi@126.com
"""
# `<path/to/cfs>`/ustb/
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
# `<path/to/cfs>`/data/
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, "../data/")) + "/"
# `<path/to/cfs>`/
PRJ_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..")) + "/"

import unittest as ut
class TestCase(ut.TestCase):
   if __name__ == '__main__':
     ut.main()

   def test_accessCf(self):
     print ROOT_DIR
     print DATA_DIR
     print PRJ_DIR
