"""
2  模型计算图搭建
2.1  文件读取
2.1.1  读取id向量
2.1.2  读取字典
2.1.3  读取词向量模型
2.2  构建词向量矩阵
2.3  统计id向量的长度，并填充为统一长度（为了批处理语料文件）
2.4  定义Tensor
2.5  Encoder
2.5.1  定义LSTM Cell
2.5.2  Embedding Layers
2.5.3  MultiRNN Cells
2.5.4  Dynamic RNN(动态RNN)
2.5.5  自定义函数
2.6  Decoder
2.6.1  添加_BOS
2.6.2  Embedding Layer
2.6.3  MultiRNN Cells
2.6.4  Projection Layer
2.6.5  Training Decoder
2.6.6  Inference Decoder
2.6.7  自定义Decoder端函数
2.7  Encoder-Decoder Model（Seq2Seq Model）
2.7.1  Loss Function
2.7.2  Optimize（自适应优化器，相比梯度下降法，速度更快）
2.7.3  梯度剪枝（防止梯度爆炸、梯度消失）
3  Train
4  Inference/Test
5  添加注意力机制后的Decoder端
"""

import json
import tensorflow as tf
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

corpus_path = r'_04_chatbot/data/dialog'
dict_file_path = r'_04_chatbot/data/mydict.txt'
model_save_path = r'_04_chatbot/data/tmp/Word2Vec.model'
dict_save_path = r'_04_chatbot/data/tmp/dict.txt'
ids_save_dict = r'_04_chatbot/data/ids/ids.json'
checkpoint_dir = r'_04_chatbot/data/checkpoints'

# 2.1  文件读取
# 2.1.1  读取id向量
with open(ids_save_dict, 'r', encoding='utf-8')as f:
	tmp = json.load(f)
fromids = tmp['fromids']
toids = tmp['toids']
del tmp, f

# 2.1.2  读取字典
with open(dict_save_path, 'r')as f:
	all_dict = f.read().split('\n')
del f
id2words = {i: j for i, j in enumerate(all_dict)}
word2ids = {j: i for i, j in enumerate(all_dict)}

# 2.1.3  读取词向量模型
model = Word2Vec.load(model_save_path)
emb_size = model.layer1_size

# 2.2  构建词向量矩阵
vocab_size = len(all_dict)
corpus_size = len(fromids)

embedding_matrix = np.zeros((vocab_size, emb_size))
tmp = np.diag([1] * emb_size)

k = 0
for i in range(vocab_size):
	try:
		embedding_matrix[i] = model.wv[str(i)]
	except:
		embedding_matrix[i] = tmp[k]
		k += 1
del k, i

# 2.3  统计id向量的长度，并填充为统一长度（为了批处理语料文件）
from_length = [len(i) for i in fromids]
m = max(from_length)
source = [i + [word2ids['_PAD']] * (m - len(i)) for i in fromids]

to_length = [len(i) for i in toids]
m = max(to_length)
target = [i + [word2ids['_PAD']] * (m - len(i)) for i in toids]
del m

# 2.4  定义Tensor
num_layers = 2  # 神经元个数
hidden_size = 100  # 隐层神经元个数
learning_rate = 0.001  # 学习率
max_inference_sequence = 35

# 输入
input_data = tf.placeholder(tf.int32, shape=[corpus_size, None], name='source')
# 输出
output_data = tf.placeholder(tf.int32, shape=[corpus_size, None], name='target')
# 输入句子的长度
input_sequence_length = tf.placeholder(tf.int32, shape=[corpus_size, ], name='source_sequence_length')
# 输出句子的长度
output_sequence_length = tf.placeholder(tf.int32, shape=[corpus_size, ], name='target_sequence_length')
# 最大输出句子的长度
max_output_sequence_length = tf.reduce_max(output_sequence_length)
# 词向量矩阵
emb_matrix = tf.constant(embedding_matrix, dtype=tf.float32, name='embedding_matrix')


# 2.5  Encoder

# 2.5.1  定义LSTM Cell
def get_lstm_cell(hidden_size):
	lstm_cell = tf.contrib.rnn.LSTMCell(
		num_units=hidden_size,
		initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=666))
	return lstm_cell


"""
# 2.5.2  Embedding Layers
encoder_embedding_input = tf.nn.embedding_lookup(params=emb_matrix, ids=input_data)
# 2.5.3  MultiRNN Cells
encoder_cells = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hidden_size) for _ in range(num_layers)])
# 2.5.4  Dynamic RNN(动态RNN)
encode_output, encoder_state = tf.nn.dynamic_rnn(
	cell=encoder_cells,
	inputs=encoder_embedding_input,
	sequence_length = input_sequence_length,
	dtype=tf.float32)
"""


# 2.5.5  自定义函数
def my_encode(emb_matrix, input_data, hidden_size, input_sequence_length):
	# 2.5.2  Embedding Layers
	encoder_embedding_input = tf.nn.embedding_lookup(params=emb_matrix, ids=input_data)
	# 2.5.3  MultiRNN Cells
	encoder_cells = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hidden_size) for _ in range(num_layers)])
	# 2.5.4  Dynamic RNN(动态RNN)
	encode_output, encoder_state = tf.nn.dynamic_rnn(
		cell=encoder_cells,
		inputs=encoder_embedding_input,
		sequence_length=input_sequence_length,
		dtype=tf.float32)
	return encode_output, encoder_state


# 2.6  Decoder
"""
# 2.6.1  添加_BOS
# 模型输入开始是有'BOS'，结束是有'EOS'，因为在测试过程中没有任何参考，生成模型
ending = tf.strided_slice(output_data, begin=[0, 0], end=[corpus_size, -1], strides=[1, 1])
begin_signal = tf.fill(dims=[corpus_size, 1], value=word2ids['_BOS'])
decoder_input_data = tf.concat([begin_signal, ending], axis=1, name='decoder_input_data')
# 2.6.2  Embedding Layer
decoder_embedding_input = tf.nn.embedding_lookup(params=emb_matrix, ids=decoder_input_data)
# 2.6.3  MultiRNN Cells
decoder_cells = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hidden_size) for _ in range(num_layers)])
# 2.6.4  Projection Layer
# Dense 是一个搭建全连接层的类
projection_layer = tf.layers.Dense(units=vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),)
# 2.6.5  Training Decoder
with tf.variable_scope('Decoder'):
	# Helper对象
	training_helper = tf.contrib.seq2seq.TrainingHelper(
		inputs=decoder_embedding_input,
		sequence_length=output_sequence_length)

	# Basic Decoder
	training_decoder = tf.contrib.seq2seq.BasicDecoder(
		cell=decoder_cells,
		helper=training_helper,
		output_layer=projection_layer,
		initial_state=decoder_cells.zero_state(batch_size=corpus_size, dtype=tf.float32))

	# Dynamic Decoder
	training_final_output, training_final_state, training_sequence_length = tf.contrib.seq2seq.dynamic_decode(
		decoder=training_decoder,
		maximum_iterations=max_output_sequence_length,
		impute_finished=True)
# 2.6.6  Inference Decoder
with tf.variable_scope('Decoder', reuse=True):
	# Helper 对象
	start_tokens = tf.tile(
		input=tf.constant(value=[word2ids['_EOS']], dtype=tf.int32),
		multiples=[corpus_size],
		name='start_tokens')
	inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
		embedding=emb_matrix,
		start_tokens=start_tokens,
		end_token=word2ids['_EOS'])

	# Basic Decoder
	inference_decoder = tf.contrib.seq2seq.BasicDecoder(
		cell=decoder_cells,
		helper=inference_helper,
		output_layer=projection_layer,
		initial_state=decoder_cells.zero_state(batch_size=corpus_size, dtype=tf.float32))

	# Dynamic Decoder
	inference_final_output, inference_final_state, inference_sequence_length = tf.contrib.seq2seq.dynamic_decode(
		decoder=inference_decoder,
		maximum_iterations=max_output_sequence_length,
		impute_finished=True)
"""


# 2.6.7  自定义Decoder端函数
def my_decoder(output_data, corpus_size, word2ids, emb_matrix, hidden_size, num_layers, vocab_size,
			   output_sequence_length, max_output_sequence_length, encoder_state):
	ending = tf.strided_slice(output_data, begin=[0, 0], end=[corpus_size, -1], strides=[1, 1])
	begin_signal = tf.fill(dims=[corpus_size, 1], value=word2ids['_BOS'])
	decoder_input_data = tf.concat([begin_signal, ending], axis=1, name='decoder_input_data')
	# 2.6.2  Embedding Layer
	decoder_embedding_input = tf.nn.embedding_lookup(params=emb_matrix, ids=decoder_input_data)
	# 2.6.3  MultiRNN Cells
	decoder_cells = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hidden_size) for _ in range(num_layers)])
	# 2.6.4  Projection Layer
	# Dense 是一个搭建全连接层的类
	projection_layer = tf.layers.Dense(units=vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
	# 2.6.5  Training Decoder
	with tf.variable_scope('Decoder'):
		# Helper对象
		training_helper = tf.contrib.seq2seq.TrainingHelper(
			inputs=decoder_embedding_input,
			sequence_length=output_sequence_length)

		# Basic Decoder
		training_decoder = tf.contrib.seq2seq.BasicDecoder(
			cell=decoder_cells,
			helper=training_helper,
			output_layer=projection_layer,
			initial_state=encoder_state)

		# Dynamic Decoder
		training_final_output, training_final_state, training_sequence_length = tf.contrib.seq2seq.dynamic_decode(
			decoder=training_decoder,
			maximum_iterations=max_output_sequence_length,
			impute_finished=True)
	# 2.6.6  Inference Decoder
	with tf.variable_scope('Decoder', reuse=True):
		# Helper 对象
		start_tokens = tf.tile(
			input=tf.constant(value=[word2ids['_EOS']], dtype=tf.int32),
			multiples=[corpus_size],
			name='start_tokens')
		inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			embedding=emb_matrix,
			start_tokens=start_tokens,
			end_token=word2ids['_EOS'])

		# Basic Decoder
		inference_decoder = tf.contrib.seq2seq.BasicDecoder(
			cell=decoder_cells,
			helper=inference_helper,
			output_layer=projection_layer,
			initial_state=encoder_state)

		# Dynamic Decoder
		inference_final_output, inference_final_state, inference_sequence_length = tf.contrib.seq2seq.dynamic_decode(
			decoder=inference_decoder,
			maximum_iterations=max_output_sequence_length,
			impute_finished=True)
		return training_final_output, training_final_state, inference_final_output, inference_final_state


# 2.7  Encoder-Decoder Model（Seq2Seq Model）
encode_output, encoder_state = my_encode(emb_matrix, input_data, hidden_size, input_sequence_length)
training_final_output, training_final_state, inference_final_output, inference_final_state = my_decoder(output_data, corpus_size, word2ids, emb_matrix, hidden_size, num_layers, vocab_size, output_sequence_length, max_output_sequence_length, encoder_state)
# 2.7.1  Loss Function
training_logits = tf.identity(input=training_final_output.rnn_output, name='training_logits')
inference_logits = tf.identity(input=inference_final_output.sample_id, name='inference_logits')
mask = tf.sequence_mask(lengths=output_sequence_length, maxlen=max_output_sequence_length, name='mask', dtype=tf.float32)
loss = tf.contrib.seq2seq.sequence_loss(logits=training_logits, targets=output_data, weights=mask)
# 2.7.2  Optimize（自适应优化器，相比梯度下降法，速度更快）
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# 2.7.3  梯度剪枝（防止梯度爆炸、梯度消失）
gradients = optimizer.compute_gradients(loss)
clip_gradients = [(tf.clip_by_value(t=grad, clip_value_max=5., clip_value_min=-5.), var) for grad, var in gradients if grad is not None]
train_op = optimizer.apply_gradients(clip_gradients)
# 3  Train
train_times = 5000
with tf.Session()as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	ckpt = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
	if ckpt is not None:
		saver.restore(sess, ckpt)
		print(f'从{checkpoint_dir}模型中读取参数'.center(50, '='))
	else:
		print('重新开始训练')

	for i in range(train_times):
		_, training_pre, loss = sess.run(
			[train_op, training_final_output.sample_id, loss],
			feed_dict={
				input_data: source,
				output_data: target,
				input_sequence_length: from_length,
				output_sequence_length: to_length
			})

		if i % 100 == 0 or i == train_times - 1:
			print(f'第{i}次训练'.center(50, '='))
			print(f'损失值为{loss}'.center(50, '='))
			print('输入:', ''.join([id2words[i] for i in source[0] if i != word2ids['_PAD']]))
			print('输出:', ''.join([id2words[i] for i in target[0] if i != word2ids['_PAD']]))
			print('Train预测:', ''.join([id2words[j] for j in training_pre[0] if j != word2ids['_PAD']]))

			inference_pre = sess.run(
				inference_final_output.sample_id,
				feed_dict={
					input_data: source,
					input_sequence_length: from_length
			})
			print(f'Inference预测:', ''.join([id2words[j] for j in inference_pre[0] if j != word2ids['_PAD']]))
			saver.save(sess, checkpoint_dir+r'/trained_model.ckpt')
			print('模型已保存'.center(50, '='))


# 4  Inference/Test
# 5  添加注意力机制后的Decoder端
