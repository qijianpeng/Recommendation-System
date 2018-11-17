#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 Created by qijianpeng on 2018/11/17.
 Email: jianpengqi@126.com
 
 Reference: https://cs.stanford.edu/~quocle/paragraph_vector.pdf
 
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import numpy as np
import gensim
import jieba
import glob
from gensim.models import Doc2Vec
import warnings
warnings.filterwarnings('ignore', '.*do not.*',)

from common.RS import RS
from utils import DATA_DIR, ROOT_DIR, PRJ_DIR

TaggededDocument = gensim.models.doc2vec.TaggedDocument

class Doc2VecBook(RS):
  def __init__(self):
    pass
  def initEnv(self):
    RS.initEnv(self)
    self.retrain = bool(raw_input("是否重新开始训练？Y/y").lower() == "y")
    self.books_path = str(raw_input("请输入书籍列表文件路径(Default newbook.txt):")
                     or (DATA_DIR + "book/newbook.txt"))
    self.stopwords_path = str(raw_input("请输入停用词文件路径(Default stopwords.txt):")
                     or (DATA_DIR + "stopwords.txt"))
    self.stopwords = set(open(self.stopwords_path).read().strip('\n').split('\n'))
    if self.retrain :
      pass
    else:
      self.model_path = str(raw_input("请输入已训练好的文件路径(Default model_test):")
                       or (ROOT_DIR + "cb/doc2vec/model_test"))

    # TODO check file if exists.

  def compute(self): #利用训练集生成Doc2Vec模型，并保存
    self.traning_data = self._corpusProcessing()
    if self.retrain:
      self.model_dm = Doc2VecBook.train(self.traning_data)
    else:
      self.model_dm = Doc2Vec.load(self.model_path)

  def recommend(self, post_desc):
    model_dm = self.model_dm
    stopwords = self.stopwords
    post_desc_words = ' '.join([x for x in jieba.lcut(post_desc) if x not in stopwords])
    # 获得对应的输入句子的向量
    inferred_vector_dm = model_dm.infer_vector(doc_words=post_desc_words)
    # 返回相似的句子
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=20)
    # 获取推荐结果
    res = ''
    for count, sim in sims:
      #sentence = self.traning_data[count].words
      #print (sentence, sim)
      book_id = str(count + 1)
      res = res + (book_id + ' ')
    return res


  def _corpusProcessing(self):
    self.books = open(self.books_path).read().strip(' ')
    print "Cutting words now..."
    self.text = ' '.join([x for x in jieba.lcut(self.books) if x not in self.stopwords])
    word_list = self.text.split('\n')
    tag_docs  = []
    for idx, doc in enumerate(word_list):
      tag_doc = TaggededDocument(doc, tags = [idx + 1])
      tag_docs.append(tag_doc)
    return tag_docs

  @classmethod
  def train(clz, traning_data, size = 200, epoch_num = 1):
    MODEL_DIR = ROOT_DIR + "ustb/cb/doc2vec/model_test"
    model_dm = Doc2Vec(traning_data, min_count = 1, window = 5, size = size,
                       sample = 1e-3, negative = 5, workers = 4, hs = 1, iter = 6)
    print "Training now..."
    model_dm.train(traning_data, total_examples=model_dm.corpus_count, epochs=100)
    model_dm.save(MODEL_DIR)
    model_dm.load
    return model_dm


import unittest as ut
class TestDoc2VecBook(ut.TestCase):
  if __name__ == '__main__':
    ut.main()

  def test_doc2vecBook(self):
    doc2vec_book =  Doc2VecBook()
    doc2vec_book.initEnv()
    doc2vec_book.compute()

    RES_DIR = ROOT_DIR + "cb/doc2vec/result.txt"
    post_files_filter = DATA_DIR + 'post/*.txt'
    post_files = glob.glob(post_files_filter)
    if os.path.exists(RES_DIR):
        os.remove(RES_DIR)
    for idx, post_file in enumerate(post_files):
      post_id = idx + 1
      print post_id
      result = open(RES_DIR, mode='a')
      t = str(post_id) + '\n'
      result.write(t)
      result.close()
      post_desc = open(post_file).read().strip(' ')
      res = doc2vec_book.recommend(post_desc)
      result.write(res)