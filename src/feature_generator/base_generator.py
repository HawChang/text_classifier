#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/23 8:32 PM
# @Author : ZhangHao
# @File   : base_generator.py
# @Desc   : 
import codecs
import jieba


class BaseFeatureGenerator(object):
	def __init__(self, word_seg_func, stopwords_path=None, encoding="utf-8", ngram=3, feature_min_length=2):
		"""
		:param word_seg_func: 切词函数选择，切词目标编码为gb18030
		"""
		self.seg_words = {
			"jieba": self.jieba_seg_words,
			"word_seg": self.baidu_seg_words
		}[word_seg_func]
		self._ngram = ngram
		self._encoding = encoding
		self._stopwords = set() if stopwords_path is None else self.load_stopword_file(stopwords_path)
		self._feature_min_length = feature_min_length
	
	def run(self, data_path):
		"""
		遍历文件记录 生成各记录的特征
		:param data_path:
		:return: 三元组列表，三元组信息：（标签，特征，其他信息）
		"""
		raise RuntimeError("BaseFeatureGenerator.run() must be overwrite.")
	
	def gen_ngram_feature(self, tar_string):
		"""
		根据字符串生成其ngram特征
		:param tar_string: 编码为unicode
		:return: 特征集合 编码为unicode
		"""
		feature_set = set()
		# 得到切词结果
		tokens = self.seg_words(tar_string)
		#print("tar string   : %s" % tar_string)
		#print("seg result   : %s" % "/ ".join(tokens))
		# 去除停用词和空白字符
		valid_tokens = [x for x in tokens if x not in self._stopwords]
		#print("valid tokens : %s" % "/ ".join(valid_tokens))
		# 生成ngram特征
		for start_pos in range(len(valid_tokens)):
			cur_feature = ""
			for offset in range(min(len(valid_tokens)-start_pos, self._ngram)):
				cur_feature += valid_tokens[start_pos + offset]
				if len(cur_feature) > self._feature_min_length:
					feature_set.add(cur_feature)
		return feature_set
	
	def jieba_seg_words(self, tar_string):
		"""
		使用结巴进行分词
		:param tar_string: 要切词的字符串 unicode编码
		:return: 返回unicode编码的切词结果列表
		"""
		# jieba分词结果是unicode编码
		return [x.strip() for x in jieba.lcut(tar_string.encode("gb18030", "ignore"))]
		
	def baidu_seg_words(self, tar_string):
		"""
		使用公司内部切词器
		:param tar_string: 要切词的字符串 unicode或gbk均可（切词前要转成gb18030格式）
		:return: 返回unicode编码的切词结果列表
		"""
		# TODO：实现功能
		pass
	
	def load_stopword_file(self, stopwords_path):
		"""
		加载停用词表
		:param stopwords_path:
		:return:
		"""
		stopwords_set = set()
		with codecs.open(stopwords_path, "r", self._encoding) as rf:
			for line in rf:
				stopwords_set.add(line.strip("\n"))
		return stopwords_set
	

if __name__ == "__main__":
	generator = BaseFeatureGenerator("jieba", "data/dict/stopwords.txt")
	tests = [u"测试是否能够正常切词",
	         u"测试一下]停用词的逻辑，是,否？能够正常切词"]
	
	for test in tests:
		feature_set = generator.gen_ngram_feature(test)
		print("/ ".join(feature_set))