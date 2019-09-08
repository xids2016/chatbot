"""
1  语料库预处理
1.1  读取语料库文件
1.2  分词，构建词典
1.3  构建映射关系
1.4  语料转为id向量
1.5  拆分成source、target
1.6  词向量训练
1.7  保存文件
"""

import os
import json
import jieba
from gensim.models import Word2Vec
from tkinter import _flatten


# 语料库预处理
class CorpusPreprocess():
	def __init__(self):
		pass

	def reading_corpus_files(self, corpus_path: str) -> list:
		"""
		# 1.1  读取语料库文件
		:param corpus_path: 语料库路径
		:return: 语料库列表
		"""
		corpus_list = os.listdir(corpus_path)
		corpus = []
		for corpus_file in corpus_list:
			with open(os.path.join(corpus_path, corpus_file), encoding='utf-8')as f:
				corpus.extend(f.readlines())
		corpus = [i.replace('\n', '') for i in corpus]
		return corpus

	def participle_build_dictionary(self, dict_file_path, corpus):
		"""
		# 1.2  分词，构建词典
		:param dict_file_path: 分词词典文件路径
		:param corpus: 语料库
		:return: 分词后的语料库列表, 词典
		"""
		jieba.load_userdict(dict_file_path)
		corpus_cut = [jieba.lcut(i) for i in corpus]
		tmp = _flatten(corpus_cut)
		_PAD, _BOS, _EOS, _UNK = '_PAD', '_BOS', '_EOS', '_UNK'
		all_dict = [_PAD, _BOS, _EOS, _UNK] + list(set(tmp))
		return corpus_cut, all_dict

	def Building_mapp_relationship(self, all_dict):
		"""
		# 1.3  构建映射关系
		:param all_dict: 词典
		:return:映射字典
		"""
		id2words = {i: j for i, j in enumerate(all_dict)}
		words2id = {j: i for i, j in enumerate(all_dict)}
		return id2words, words2id

	def corpus_converted_to_id_vector(self, words2id, corpus_cut):
		"""
		# 1.4  语料转为id向量
		:return:
		"""
		cor2vec = [[words2id.get(word, words2id['_UNK']) for word in line]for line in corpus_cut]
		return cor2vec

	def div_to_source_and_target(self, cor2vec):
		"""
		# 1.5  拆分成source、target
		:param cor2vec: id向量
		:return: 问题和回答向量
		"""
		fromids = cor2vec[::2]
		toids = cor2vec[1::2]
		return fromids, toids

	def train_wordvec(self, cor2vec, model_save_path):
		"""
		# 1.6  词向量训练
		:param cor2vec: id向量
		:param model_save_path: 模型保存路径
		:return:None
		"""
		emb_size = 50
		temp = [list(map(str, i)) for i in cor2vec]
		if not os.path.exists(model_save_path):
			model = Word2Vec(temp, size=emb_size, window=10, min_count=1, workers=4)
			model.save(model_save_path)
		else:
			print(f'模型已经构建，可以直接调取'.center(50, '='))
		return None

	def save_file(self, dict_save_path, ids_save_path, formids, toids, all_dict):
		"""
		# 1.7  保存文件
		fromids, toids, dict
		:return:None
		"""
		with open(ids_save_path)as f:
			json.dump({'formids': formids, 'toids': toids}, fp=f, ensure_ascii=False)
		with open(dict_save_path, 'w') as f:
			f.write('\n'.join(all_dict))

	def _run(self):
		corpus = self.reading_corpus_files(corpus_path)
		_, all_dict = self.participle_build_dictionary(dict_file_path, corpus)
		my_dict = self.Building_mapp_relationship(all_dict)
		return my_dict


if __name__ == '__main__':
	corpus_path = r'_04_chatbot/data/dialog'
	dict_file_path = r'_04_chatbot/data/mydict.txt'
	model_save_path = r'_04_chatbot/data/tmp/Word2Vec.model'
	dict_save_path = r'_04_chatbot/data/tmp/dict.txt'
	ids_save_dict = r'_04_chatbot/data/ids/ids.json'

