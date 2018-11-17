#!/usr/bin/python
# -*- coding: UTF-8 -*-



from utils import ROOT_DIR
from utils import DATA_DIR
import sys
import gensim
import jieba
import numpy as np
from jieba import analyse
import os
import glob

from gensim.models.doc2vec import Doc2Vec, LabeledSentence

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest(): #读取15998本图书的书名，并进行分词，产生训练集
    fin = open(DATA_DIR + "book/newbook.txt",encoding='utf8').read().strip(' ')   #strip()去除首尾空格
    # 添加自定义的词库用于分割或重组模块不能处理的词组。
    # jieba.load_userdict("userdict.txt")
    # print(jieba.lcut(fin))
    # 添加自定义的停用词库，去除句子中的停用词。
    stopwords = set(open(DATA_DIR + 'stopwords.txt',encoding='utf8').read().strip('\n').split('\n'))   #读入停用词
    text = ' '.join([x for x in jieba.lcut(fin) if x not in stopwords])  #分词，去掉停用词中的词
    # print(text)
    # print (type(text),len(text))
    x_train = []
    word_list = text.split('\n')
    #print(word_list[0])
    # print(word_list)
    # print(type(word_list), len(word_list))
    for i,sub_list in enumerate(word_list):
        document = TaggededDocument(sub_list, tags=[i+1])
        # document是一个Tupple,形式为：TaggedDocument( 杨千嬅 现在 教育 变成 一种 生意 , [42732])
        #print(document)
        x_train.append(document)
    return x_train
'''
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)
'''
def train(x_train, size=200, epoch_num=1): #利用训练集生成Doc2Vec模型，并保存
    # D2V参数解释：
    # min_count：忽略所有单词中单词频率小于这个值的单词。一般为5，语料库少时可设置小一些
    # window：窗口的尺寸。（句子中当前和预测单词之间的最大距离）
    # size:特征向量的维度
    # sample：高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
    # negative: 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）。默认值是5。
    # workers：用于控制训练的并行数。
    model_dm = Doc2Vec(x_train,min_count=1, window = 5, size = size, sample=1e-3, negative=5, workers=4,hs=1,iter=6)
    # total_examples：统计句子数
    # epochs：在语料库上的迭代次数(epochs)。
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=100)
    model_dm.save('model_test')#训练好的模型，保存
    model_dm.load
    return model_dm

def test(filename): #对每个岗位的说明分词后作为测试集，放入训练好的模型中，返回相似文本
    #加载训练好的模型
    model_dm = Doc2Vec.load("model_test")
    gang = open(filename, encoding='utf8').read().strip(' ')
    stopwords = set(open(DATA_DIR + 'stopwords.txt',encoding='utf8').read().strip('\n').split('\n'))
    #去掉停用词中的词
    test_text = ' '.join([x for x in jieba.lcut(gang) if x not in stopwords])
    # print(test_text)
    #获得对应的输入句子的向量
    inferred_vector_dm = model_dm.infer_vector(doc_words=test_text)
    # print(inferred_vector_dm)
    #返回相似的句子
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=20)
    return sims


if __name__ == '__main__':

    x_train = get_datasest()
    # print(x_train)
    # print(type(x_train), len(x_train))
    #训练模型
    model_dm = train(x_train)
    files = glob.glob(DATA_DIR + '/post\*.txt') #读取岗位说明文件，共计293个
    # files = glob.glob('ceshi\*.txt')
    # all = codecs.open('all.txt', 'a')
    if os.path.exists('result.txt'):
        os.remove('result.txt')
    i=1
    for filename in files:
        print(i) #输出岗位编号，从1开始递增
        result=open('result.txt',mode='a',encoding='utf-8')
        postID=str(i)
        result.write(postID)
        result.write('\n')
        result.close()
        sims=test(filename)
        for count, sim in sims:
            # print(count+1)
            sentence = str(x_train[count])
            # sentence = x_train[count]
            # print('sentence:'+sentence)
            # print('sim:'+str(sim))
            # print(sentence, sim, len(sentence))
            print(sentence,sim) #输出推荐图书的书名分词以及相似度
            bookID=str(count+1)
            result = open('result.txt', mode='a',encoding='utf-8')
            result.write(bookID)
            result.write(' ')
            result.close()
        i=i+1
        result = open('result.txt', mode='a', encoding='utf-8')
        result.write('\n')
        result.close()