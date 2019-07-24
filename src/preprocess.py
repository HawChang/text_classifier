#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/23 8:36 PM
# @Author : ZhangHao
# @File   : preprocess.py.py
# @Desc   : 为原始数据集生成特征 并划分训练集和数据集

import os
import codecs
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time

import config
from utils.logger import Logger

log = Logger().get_logger()


class Preprocessor(object):
	def __init__(self, encoding="utf-8"):
		self.feature_generator = \
			config.data_feature_generator(
					config.word_seg_func,
					config.stopwords_path,
					encoding=encoding,
					feature_min_length=config.feature_min_length)
		self.encoding = encoding
	
	def run(self):
		"""
		为原始数据集生成特征 并划分训练集和数据集
		:return: None
		"""
		if config.re_gen_feature:
			log.info("gen feature...")
			start_time = time.time()
			self.gen_feature()
			log.info("gen feature end. cost time = %.4fs" % (time.time() - start_time))
		
		if config.re_split_train_test:
			log.info("split train test data...")
			start_time = time.time()
			self.data_train_test_split()
			log.info("split train test data. cost time = %.4fs" % (time.time() - start_time))
		
	def gen_feature(self):
		"""
		为训练数据各记录生成特征
		:return: None
		"""
		data_feature_list = list()
		# 首先检查原始数据是文件还是文件夹
		if os.path.isdir(config.origin_data):
			
			files = os.listdir(config.origin_data)
			# 遍历文件夹中的每一个文件
			for file_name in files:
				# 如果文件名以.开头 说明该文件隐藏 不是正常的数据文件
				if file_name[0] == ".":
					continue
				file_path = os.path.join(config.origin_data, file_name)
				log.info("process origin data : %s" % file_path)
				data_feature_list.extend(self.feature_generator.run(file_path))
		elif os.path.isfile(config.origin_data):
			log.info("process origin data : %s" % config.origin_data)
			data_feature_list = self.feature_generator.run(config.origin_data)
			pass
		else:
			raise TypeError("unknown origin data : %s." % config.origin_data)
		
		# 存入文件
		# 文件中数据有三列 标签、特征、额外信息
		with codecs.open(config.origin_data_feature_path, "w", self.encoding) as wf:
			for data in data_feature_list:
				wf.write(config.col_sep.join(data) + "\n")
		
	def data_train_test_split(self):
		"""
		根据data_list按标签比例随机划分训练测试集
		:return: None
		"""
		label_data_dict = defaultdict(list)
		# 读入所有数据 以label为key存入字典
		with codecs.open(config.origin_data_feature_path, "r", self.encoding) as rf:
			for line in rf:
				parts = line.strip('\n').split(config.col_sep)
				label_data_dict[parts[0]].append(parts)
		# 各label按指定比例划分训练测试集
		train_data = list()
		test_data = list()
		for label, data_list in label_data_dict.items():
			cur_train_data, cur_test_data = train_test_split(data_list, test_size=config.test_ratio)
			train_data.extend(cur_train_data)
			test_data.extend(cur_test_data)
		
		# 存入文件
		# 原始数据文件
		with codecs.open(config.train_data_path, "w", self.encoding) as wf:
			wf.writelines("\n".join([config.col_sep.join(x) for x in train_data]))
		with codecs.open(config.test_data_path, "w", self.encoding) as wf:
			wf.writelines("\n".join([config.col_sep.join(x) for x in test_data]))
		# 训练文件 只保留标签和特征
		with codecs.open(config.train_data_feature_path, "w", self.encoding) as wf:
			wf.writelines("\n".join([config.col_sep.join(x[:2]) for x in train_data]))
		with codecs.open(config.test_data_feature_path, "w", self.encoding) as wf:
			wf.writelines("\n".join([config.col_sep.join(x[:2]) for x in test_data]))
		

if __name__ == "__main__":
	preprocessor = Preprocessor()
	preprocessor.run()