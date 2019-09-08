import tensorflow as tf
import jieba
from gensim.models import Word2Vec


# 读取字典、词向量矩阵
with open('./tmp/dic.txt', 'r') as f:
    all_dict = f.read().split('\n')
words2id = {j: i for i, j in enumerate(all_dict)}
id2words = {i: j for i, j in enumerate(all_dict)}

model = Word2Vec.load('./tmp/word2vec.model')
emb_size = model.layer1_size

# 计算图调用
tf.reset_default_graph()
corpus_size = 62
loaded_graph = tf.Graph()  # TensorFlow计算图

checkpoint = './checkpoints/trained_model.ckpt'  # checkpoint文件保存

loaded_graph = tf.Graph()  # TensorFlow计算，表示为数据流图。
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    #    .meta文件保存了当前图结构
    #    .index文件保存了当前参数名
    #    .data文件保存了当前参数值
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('source:0')
    logits = loaded_graph.get_tensor_by_name('inference_logits:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    emb_matrix = loaded_graph.get_tensor_by_name('embedding_matrix:0')

    # 调用计算图进行测试

    print('让我开始聊天吧！\nps:输入bye停止。')
    while True:
        input_words = input('>>> ')  # ‘你好’

        if input_words.lower() == 'bye':
            break

        input_ids = [words2id.get(i, words2id['_UNK']) for i in jieba.lcut(input_words)]  # [114]

        answer_logits = sess.run(logits, feed_dict={input_data: [input_ids] * corpus_size,
                                                    source_sequence_length: [len(input_ids)] * corpus_size})[0]

        print(''.join([id2words[i] for i in answer_logits if i != words2id['_PAD']]))