#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/24 11:57 AM
# @Author : ZhangHao
# @File   : logger.py
# @Desc   : 

import logging


class Logger(object):
	_is_init = False
	
	def __init__(self):
		if not self._is_init:
			logging.basicConfig(
				#filename="log/run.log",
				level=logging.DEBUG,
				format="[%(asctime)s][%(filename)s:%(funcName)s:%(lineno)s][%(levelname)s]:%(message)s",
				datefmt='%Y-%m-%d %H:%M:%S')
			#ch = logging.StreamHandler()
			self.logger = logging.getLogger()
			#self.logger.addHandler(ch)
			self._is_init = True
	
	def get_logger(self):
		return self.logger


if __name__ == "__main__":
	pass