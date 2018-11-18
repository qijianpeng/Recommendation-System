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
from environments import DATA_DIR, ROOT_DIR, PRJ_DIR

TaggededDocument = gensim.models.doc2vec.TaggedDocument

class Doc2VecBook(RS):
  def __init__(self):
    pass
  def initEnv(self):
    RS.initEnv(self)
    self.retrain = bool(raw_input("是否重新开始训练？Y/y").lower() == "y")
    self.books_path = str(raw_input("请输入书籍列表文件路径(Default newbook.txt):")
                     or (DATA_DIR + "book/newbook.txt"))
    # 添加自定义的词库用于分割或重组模块不能处理的词组。
    # jieba.load_userdict("userdict.txt")
    # 添加自定义的停用词库，去除句子中的停用词。
    self.stopwords_path = str(raw_input("请输入停用词文件路径(Default stopwords.txt):")
                     or (DATA_DIR + "stopwords.txt"))
    self.stopwords = set(open(self.stopwords_path).read().strip('\n').split('\n'))
    if self.retrain :
      pass
    else:
      self.model_path = str(raw_input("请输入已训练好的文件路径(Default model_test):")
                       or (ROOT_DIR + "cb/doc2vec/model_test"))

    # TODO check file if exists.

  def compute(self):# 用于获取doc2vec模型
    self._corpusProcessing() # 语料处理
    if self.retrain:
      # 利用训练集生成Doc2Vec模型，并保存
      self.model_dm = Doc2VecBook.train(self.traning_data)
    else:
      # 加载已生成的模型
      self.model_dm = Doc2Vec.load(self.model_path)

  def recommend(self, post_desc, topn = 20):
    """根据`post_desc`推荐`top-n`的item(id).
    
    :param post_desc: 指定岗位的描述文本内容
    :return: 
    """
    model_dm = self.model_dm
    stopwords = self.stopwords
    post_desc_words = ' '.join([x for x in jieba.lcut(post_desc) if x not in stopwords])
    # 获得对应的输入句子的向量
    inferred_vector_dm = model_dm.infer_vector(doc_words=post_desc_words)
    # 返回相似的句子
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=topn)
    # 获取推荐结果
    res = ''
    for count, sim in sims:
      #sentence = self.traning_data[count].words
      #print (sentence, sim)
      book_id = str(count + 1)
      res = res + (book_id + ' ')
    return res


  def _corpusProcessing(self):# 语料处理，对书籍分词及打tag
    self.books = open(self.books_path).read().strip(' ')
    print "Cutting words now..."
    self.text = ' '.join([x for x in jieba.lcut(self.books) if x not in self.stopwords])
    word_list = self.text.split('\n')
    tag_docs  = []
    for idx, doc in enumerate(word_list):
      tag_doc = TaggededDocument(doc, tags = [idx + 1])
      tag_docs.append(tag_doc)
    self.traning_data = tag_docs

  @classmethod
  def train(clz, traning_data, size = 200, epoch_num = 1):
    MODEL_DIR = ROOT_DIR + "ustb/cb/doc2vec/model_test"

    # D2V参数解释：
    # min_count：忽略所有单词中单词频率小于这个值的单词。一般为5，语料库少时可设置小一些
    # window：窗口的尺寸。（句子中当前和预测单词之间的最大距离）
    # size:特征向量的维度
    # sample：高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
    # negative: 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）。默认值是5。
    # workers：用于控制训练的并行数。
    model_dm = Doc2Vec(traning_data, min_count = 1, window = 5, size = size,
                       sample = 1e-3, negative = 5, workers = 4, hs = 1, iter = 6)
    print "Training now..."
    # total_examples：统计句子数
    # epochs：在语料库上的迭代次数(epochs)。
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