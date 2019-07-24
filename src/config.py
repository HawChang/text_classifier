#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/23 8:04 PM
# @Author : ZhangHao
# @File   : config.py
# @Desc   : 通用配置
import os
import feature_generator.base_generator
import feature_generator.card_qualification_data

# 原始训练数据总目录
origin_data = "data/origin_data"

# 处理原始数据 生成特征的类
data_feature_generator = feature_generator.card_qualification_data.CardDataFeatureGenerator

# word_seg_type
# 切词可选:结巴"jieba" 和 公司内部切词类"word_seg"
word_seg_func = "jieba"

# 特征的最小长度
feature_min_length = 2

# 停用词
stopwords_path = "data/dict/stopwords.txt"

# 每行记录添加其生成的特征
# 总体生成特征后 再划分训练集和验证集
origin_data_feature_path = "data/origin_data_feature"

# 划分训练集和测试集
test_ratio = 0.1

# 训练集和测试集
# train.txt 是训练数据集的全部信息 包括类别、特征、原数据等
# train_feature.txt 是train.txt的精简，只包括类别和特征两列，用于训练
# 测试文件同理
train_data_path = "data/train.txt"
train_data_feature_path = "data/train_feature.txt"
test_data_path = "data/test.txt"
test_data_feature_path = "data/test_feature.txt"

# 数据集各列之间的间隔符
col_sep = "\t"

# 是否要重新生成特征
re_gen_feature = True

# 是否要重新划分训练测试集
re_split_train_test = True

# 是否是测试
is_debug = True

# 可选："logistic_regression"
model = "logistic_regression"

# 可选："tf_word"
feature_type = "tf_word"

# 特征频率阈值 小于该频率的特征省略
feature_min_df = 2

# vectorizer读取特征时用的token_pattern
token_pattern = r'(?u)[^ ]+'

# 可选 "l1"、"l2"
penalty = "l1"

# lr参数
# 正则化系数 可以是int,float或list
Cs = [0.1,0.2,0.5,1,2,5,10,20]

# 各类别的权重
class_weight = {
	0: 1,
	1: 1
}

# 交叉校验确定系数C的fold数
k_fold = 5


# 输出文件地址
output_dir = "output"
vectorizer_path = output_dir + "/vectorizer_" + feature_type + "_" + model + ".pkl"
model_path = output_dir + "/model_" + feature_type + "_" + model + ".pkl"
feature_weight_path = output_dir + "/feature_weight_" + feature_type + "_" + model + ".txt"
label_vocab_path = output_dir + "/label_" + feature_type + "_" + model + ".txt"

# 测试文件地址
predict_dir = output_dir + "/pred_" + feature_type + "_" + model + ".txt"


if not os.path.exists(output_dir):
	os.mkdir(output_dir)



