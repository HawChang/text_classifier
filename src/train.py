#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/24 2:39 PM
# @Author : ZhangHao
# @File   : train.py
# @Desc   : 

import os
import pickle
import codecs
import config
import time
import numpy as np
from utils.logger import Logger
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegressionCV

log = Logger().get_logger()


class Trainer(object):
	def __init__(self):
		self.train_method = {
			"logistic_regression" : self.logistic_regression
		}[config.model]
		
		self.vectorizer = {
			"tf_word": CountVectorizer(
				analyzer="word",
				encoding="utf-8",
				lowercase=True,
				min_df=config.feature_min_df,
				token_pattern=config.token_pattern)
		}[config.feature_type]
	
	def run(self):
		log.info("train start...")
		start_time = time.time()
		self.train_method()
		log.info("train end. cost time = %.4fs" % (time.time() - start_time))
	
	def logistic_regression(self):
		label_vec, data_vec = self.gen_vec(config.train_data_feature_path)
		log.info("label occur : %s" % ",".join([str(label) for label in set(label_vec)]))
		solver = "liblinear" if config.penalty == "l1" else "lbfgs"
		log.info("model train start...")
		start_time = time.time()
		model = LogisticRegressionCV(
			refit=True,
			n_jobs=-1,
			solver=solver,
			Cs=config.Cs,
			fit_intercept=False,
			class_weight=config.class_weight,
			penalty=config.penalty,
			cv=config.k_fold).fit(data_vec, label_vec)
		log.info("model train end. cost time = %.4fs" % (time.time() - start_time))
		
		log.info("save model start...")
		start_time = time.time()
		self.dump_pkl(model, config.model_path)
		log.info("save model end. cost time = %.4fs" % (time.time() - start_time))
		
		log.info("model score : %f" % model.score(data_vec, label_vec))
		#print(model.score(data_vec, label_vec))
		
		log.info("save model start...")
		start_time = time.time()
		with codecs.open(config.feature_weight_path, "w", "utf-8") as wf: # todo:只是二分类的存储 后续变成多分类
			# 如果是二分类 则model.coef_形状为(1, n_features) 否则为(n_classes, n_features)
			for feature, weight in sorted(zip(self.vectorizer.get_feature_names(), model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
				if weight == 0:
					continue
				wf.write("%s\t%f" % (feature, weight) + "\n")
		log.info("save model end. cost time = %.4fs" % (time.time() - start_time))
	
	def gen_vec(self, data_path):
		"""
		根据数据文件生成矩阵
		数据文件格式：标签，特征
		:return:label_vec 和 data_vec
		"""
		label_list = list()
		data_list = list()
		with codecs.open(data_path, "r", "utf-8") as rf:
			for line in rf:
				parts = line.strip("\n").split(config.col_sep)
				label_list.append(parts[0])
				data_list.append(parts[1])
		
		log.info("data to vec start...")
		start_time = time.time()
		# 转vec
		label_vec = np.array(label_list,dtype=np.int32)
		data_vec = self.vectorizer.fit_transform(data_list)
		log.info("data to vec end. cost time = %.4fs" % (time.time() - start_time))
		log.info("train data vec shape :" + str(data_vec.shape))
		
		for word, freq in self.vectorizer.vocabulary_.items()[:20]:
			log.debug("%s\t:%d" %(word, freq))
		
		log.info("save vectorizer...")
		start_time = time.time()
		self.dump_pkl(self.vectorizer, config.vectorizer_path)
		log.info("save vectorizer end. cost time = %.4fs" % (time.time() - start_time))
		return label_vec, data_vec
	
	def dump_pkl(self, obj, pkl_path, overwrite=True):
		if not pkl_path:
			log.warning("file(%s) name wrong." % pkl_path)
		if os.path.exists(pkl_path) and not overwrite:
			log.warning("file(%s) exist and not overwrite." % pkl_path)
		if pkl_path:
			with open(pkl_path, "wb") as wf:
				pickle.dump(obj, wf)
			log.info("save to %s" % pkl_path)
			

if __name__ == "__main__":
	trainer = Trainer()
	trainer.run()
