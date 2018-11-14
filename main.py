import tensorflow as tf
from utils import get_data_info, read_data, load_word_embeddings, get_batch_index
from tensorflow.python.saved_model import tag_constants
from model import IAN
import os
import time
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
word2id_file = open('word2id.obj','rb')
train_data_file = open('train_data.obj','rb')
test_data_file = open('test_data.obj','rb')
embedding_file = open('embedding_file.obj','rb')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_epoch', 10, 'number of epoch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('pre_processed', 1, 'Whether the data is pre-processed')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

tf.app.flags.DEFINE_string('embedding_file_name', 'data/glove.840B.300d.txt', 'embedding file name')
tf.app.flags.DEFINE_string('dataset', 'data/restaurant/', 'the directory of dataset')

tf.app.flags.DEFINE_integer('max_aspect_len', 0, 'max length of aspects')
tf.app.flags.DEFINE_integer('max_context_len', 0, 'max length of contexts')
tf.app.flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')


def main(_):
    start_time = time.time()

    #print('Loading data info ...')
    #word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(FLAGS.dataset, FLAGS.pre_processed)
    #print(FLAGS.max_aspect_len, FLAGS.max_context_len)
    #pickle.dump(word2id,word2id_file)


    #print('Loading training data and testing data ...')
    #train_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, FLAGS.dataset + 'train', FLAGS.pre_processed)
    #test_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, FLAGS.dataset + 'test', FLAGS.pre_processed)
    #pickle.dump(train_data,train_data_file)
    #pickle.dump(test_data,test_data_file)

    #print('Loading pre-trained word vectors ...')
    #FLAGS.embedding_matrix = load_word_embeddings(FLAGS.embedding_file_name, FLAGS.embedding_dim, word2id)
    #pickle.dump(FLAGS.embedding_matrix,embedding_file)

    train_data=pickle.load(train_data_file)
    test_data=pickle.load(test_data_file)
    FLAGS.max_aspect_len=23
    FLAGS.max_context_len=78
    print('loading embeddings')
    FLAGS.embedding_matrix = pickle.load(embedding_file)

    with tf.Session() as sess:
        #model = IAN(FLAGS, sess)
        #model.build_model()
        #model.run(train_data, test_data)
        print('predicting')
        predict_sentiment(sess, test_data)

    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))
    word2id_file.close()
    train_data_file.close()
    test_data_file.close()
    embedding_file.close()

def predict_sentiment(sess, test_data):
    tf.saved_model.loader.load(sess,[tag_constants.SERVING],'models2/iter_0')

    graph = tf.get_default_graph()
    for v in tf.get_default_graph().as_graph_def().node:
        print(v.name)
    selfaspects=graph.get_tensor_by_name('inputs/aspects:0')
    selfcontexts=graph.get_tensor_by_name('inputs/contexts:0')
    selflabels=graph.get_tensor_by_name('inputs/labels:0')
    selfaspect_lens=graph.get_tensor_by_name('inputs/aspect_lens:0')
    selfcontext_lens=graph.get_tensor_by_name('inputs/context_lens:0')
    selfdropout_keep_prob=graph.get_tensor_by_name('inputs/dropout_keep_prob:0')

    selfcost=graph.get_tensor_by_name('loss/cost:0')
    selfaccuracy=graph.get_tensor_by_name('predict/accuracy:0')
    selfpredict=graph.get_tensor_by_name('dynamic_rnn/predict_id:0')

    cost, acc, cnt = 0., 0, 0
    aspects, contexts, labels, aspect_lens, context_lens = test_data
    with open('predict/test_demo.txt', 'w') as f:
        for sample, num in get_batch_data(selfaspects, selfcontexts, selflabels, selfaspect_lens, selfcontext_lens, selfdropout_keep_prob, aspects, contexts, labels, aspect_lens, context_lens, len(aspects), False, 1.0):
            loss, accuracy, predict = sess.run([selfcost, selfaccuracy, selfpredict], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num
            for pred in predict:
            	f.write('%s\n' % (str(pred),))

        f.write('%f\n%f\n' % (acc/cnt,cost/cnt))
    print('Finishing analyzing testing data')

#def predict_sentiment2(sess, test_data):
#    new_saver = tf.train.import_meta_graph('models/model_final.meta')
#    new_saver.restore(sess, tf.train.latest_checkpoint('./models'))
#    timestamp = str(int(time.time()))
#    cost, acc, cnt = 0., 0, 0
#
#    graph = tf.get_default_graph()
#    for v in graph.as_graph_def().node:
#        print(v.name)
#    #selfaspects=graph.get_tensor_by_name('aspects:0')
#    #selfcontexts=graph.get_tensor_by_name('contexts:0')
#    #selflabels=graph.get_tensor_by_name('labels:0')
#    #selfaspect_lens=graph.get_tensor_by_name('aspect_lens:0')
#    #selfcontext_lens=graph.get_tensor_by_name('context_lens:0')
#    #selfdropout_keep_prob=graph.get_tensor_by_name('dropout_keep_prob:0')
#
#    selfaspects = tf.placeholder(tf.int32, [None, 23])
#    selfcontexts = tf.placeholder(tf.int32, [None, 78])
#    selflabels = tf.placeholder(tf.int32, [None, 3])
#    selfaspect_lens = tf.placeholder(tf.int32, None)
#    selfcontext_lens = tf.placeholder(tf.int32, None)
#    selfdropout_keep_prob = tf.placeholder(tf.float32)
#
#    selfloss=graph.get_tensor_by_name('loss_1:0')
#    selfaccuracy=graph.get_tensor_by_name('acc:0')
#
#    aspects, contexts, labels, aspect_lens, context_lens = test_data
#    with open('predict/test_' + str(timestamp) + '.txt', 'w') as f:
#        for sample, num in get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(aspects), False, 1.0):
#            loss, accuracy = sess.run([selfloss, selfaccuracy], feed_dict=sample)
#            cost += loss * num
#            acc += accuracy
#            cnt += num
#
#    f.write('%f\n%f\n' % (acc,cost))
#    print('Finishing analyzing testing data')

def get_batch_data(selfaspects, selfcontexts, selflabels, selfaspect_lens, selfcontext_lens, selfdropout_keep_prob, aspects, contexts, labels, aspect_lens, context_lens, batch_size, is_shuffle, keep_prob):
    for index in get_batch_index(len(aspects), batch_size, is_shuffle):
        feed_dict = {
            selfaspects: aspects[index],
            selfcontexts: contexts[index],
            selflabels: labels[index],
            selfaspect_lens: aspect_lens[index],
            selfcontext_lens: context_lens[index],
            selfdropout_keep_prob: keep_prob,
        }
        yield feed_dict, len(index)

if __name__ == '__main__':
    tf.app.run()
