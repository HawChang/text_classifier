#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/23 8:31 PM
# @Author : ZhangHao
# @File   : card_qualification_data.py
# @Desc   : 为棋牌无资质数据生成特征

import base_generator
import codecs
from utils.logger import Logger
log = Logger().get_logger()


class CardDataFeatureGenerator(base_generator.BaseFeatureGenerator):
    def __init__(self, word_seg_func, stopwords_path=None, encoding="utf-8", ngram=3, feature_min_length=2):
        base_generator.BaseFeatureGenerator.__init__(self, word_seg_func, stopwords_path, encoding, ngram, feature_min_length)
        self.encoding = encoding
        
    def run(self, data_path):
        """
        遍历文件记录 生成各记录的特征
        :param data_path:
        :param encoding:
        :return: 列表，列表元素三元组信息：（标签,特征,额外信息（原数据、userid等））
        """
        label_data_list = list()
        with codecs.open(data_path, "r", self.encoding) as rf:
            for index, line in enumerate(rf):
                parts = line.strip("\n").split('\t')
                # 棋牌数据有三列  标签、物料、userid
                label = parts[0]
                contents = parts[1].split("||")
                info = "\t".join(parts[1:])
                # 获取该记录所有特征
                feature_set = set()
                for content in contents:
                    feature_set |= self.gen_ngram_feature(content)
                label_data_list.append((label, " ".join(feature_set), info))
                if index % 10000 == 0:
                    log.info("process line #%d" % index)
                    log.info("content : %s" % parts[1])
                    log.info("="*100)
                    log.info("seg result : %s" % " ".join(feature_set))
        return label_data_list
    

if __name__ == "__main__":
    generator = CardDataFeatureGenerator("jieba")
    data_list = generator.run("../../data/origin_data/test.txt")
    for data in data_list:
        print("\t".join(data).encode("utf-8"))

