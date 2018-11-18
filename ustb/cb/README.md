# Books recommendation based Doc2Vec
Recommending books given post description.

## Requirements:
- [Gensim 3.6.0](http://radimrehurek.com/gensim): Python framework for
fast Vector Space Modelling;
- [Numpy 1.13](http://www.numpy.org) (notes: gensim 3.6.0 not compatible for numpy version 1.14 and
  above, may be you need downgrade your numpy version for temporary.)
- [jieba 0.39](https://github.com/fxsjy/jieba): Chinese Words
Segementation Utilities.

## Demo Capture

![Doc2Vec Demo](doc2vec_demo.png)

## Structure

### Datasets
  - [newbook.txt](../../data/book/newbook.txt): Books library for training.
  - [post/*.txt](../../data/post): Posts description.
  - [stopwords.txt](../../data/stopwords.txt): Stop-words for cutting text.

### Framework

`Doc2VecBook` steps:

1. `initEnv`: For inputting data files.
2. `compute`: Training model of Doc2Vec.
3. `recommend`: Making recommendations given the post description.
4. `showResult`: Writing results to observable devices.
