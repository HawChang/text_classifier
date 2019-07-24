#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/24 5:51 PM
# @Author : ZhangHao
# @File   : predict.py
# @Desc   :
import pickle
import codecs
import config
import math
from sklearn.metrics import classification_report

class Predict(object):
	def __init__(self, is_pkl=False):
		#self.model = self.load_pkl(config.model_path) if is_pkl else self.load_feature_weight(config.feature_weight_path)
		self.feature_weight_model = self.load_feature_weight(config.feature_weight_path)
		self.pkl_model = self.load_pkl(config.model_path)
		self.vectorizer = self.load_pkl(config.vectorizer_path)
		#self.predict = {
		#	True : self.pkl_predict,
		#	False: self.feature_weight_predict
		#}[is_pkl]
		self.is_pkl = is_pkl
	
	def predict(self):
		label_list = list()
		label_pred_list = list()
		total_num = 0
		wrong_num = 0
		with codecs.open(config.test_data_feature_path, "r", "utf-8") as rf:
			for line in rf:
				parts = line.strip("\n").split(config.col_sep)
				label = parts[0]
				label_list.append(label)
				data = parts[1]
				# print(line.strip("\n"))
				# print self.pkl_predict(data)
				prob, evidence = self.feature_weight_predict(data)
				hit_feature_num = len(evidence)
				label_pred = "0" if prob < config.confidence or hit_feature_num < config.min_hit_feature else "1"
				evidence_str = "||".join(["%s(%.4f)" % (word, weight) for word, weight in evidence])
				
				label_pred_list.append(label_pred)
				if label != label_pred:
					print("\t".join([label_pred, str(prob), evidence_str, line.strip("\n")]))
					wrong_num += 1
				total_num += 1
	
		print(classification_report(label_list, label_pred_list, target_names=[u"无风险", u"有风险"]))
		print("total num = %d" % total_num)
		print("wrong num = %d" % wrong_num)
	
	def feature_weight_predict(self, tar_string):
		evidence = list()
		sum = 0.0
		for word in tar_string.split(" "):
			if word not in self.feature_weight_model:
				continue
			cur_weight = self.feature_weight_model[word]
			sum += cur_weight
			evidence.append((word, cur_weight))
		
		prob = 1.0 / (1.0 + math.exp(-sum))
		top_evidence = sorted(evidence, key=lambda x:abs(x[1]), reverse=True)[:config.evidence_num]
		return prob, top_evidence
	
	def pkl_predict(self, tar_string):
		data_vec = self.vectorizer.transform([tar_string])
		res = self.pkl_model.predict_proba(data_vec)
		return res
	
	def load_pkl(self, model_path):
		with open(model_path, "rb") as rf:
			return pickle.load(rf)
	
	def load_feature_weight(self, model_path):
		feature_weight_dict = dict()
		with codecs.open(model_path, "r", "utf-8") as rf:
			for line in rf:
				parts = line.strip("\n").split("\t")
				feature_weight_dict[parts[0]] = float(parts[1])
		return feature_weight_dict
		
if __name__ == "__main__":
	predict = Predict()
	predict.predict()