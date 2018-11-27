from utils import *
from ed_model import *
import numpy as np
import tensorflow as tf
import os, datetime, time, pickle
from sklearn.metrics import precision_recall_fscore_support


tf.flags.DEFINE_float("split", 0.7, "dmm")
tf.flags.DEFINE_float("dev_size", 0.15,"dmm")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("evaluate_every", 100, "")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "")
tf.flags.DEFINE_integer("num_epochs", 300, "")
FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
    vectors, sents, anchor = load_data("windows1.bin", "labels1.bin")
    _, sents_test1, anchor_test1 = load_data("windows2.bin", "labels2.bin")
    '''
    _, sent_test2, anchor_test2 = load_data("windows3.bin", "labels3.bin")
    _, sent_test3, anchor_test3 = load_data("windows4.bin", "labels4.bin")
    '''
    sents = np.array(sents)
    anchor = np.array(anchor)
    vocab_length = len(vectors)
    print(len(sents))
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(sents)))
    sent_shuffled = sents[shuffle_indices]
    anchor_shuffled = anchor[shuffle_indices]

    dev_sample_index = int(FLAGS.split * float(len(sents)))
    test_sample_index = dev_sample_index + int(FLAGS.dev_size * float(len(sents)))
    sent_train, sent_dev, sent_test = sent_shuffled[:dev_sample_index], sent_shuffled[dev_sample_index: test_sample_index],\
                                      sent_shuffled[test_sample_index:]
    anchor_train, anchor_dev, anchor_test = anchor_shuffled[:dev_sample_index], anchor_shuffled[dev_sample_index: test_sample_index], \
                               anchor_shuffled[test_sample_index:]
    sent_dev, anchor_dev = data_evaluate(sent_dev, anchor_dev)
    sent_test, anchor_test = data_evaluate(sent_test, anchor_test)
    anchor_train_std = np.zeros((len(anchor_train), 34))
    anchor_train_std[range(len(anchor_train)), anchor_train] = 1
    anchor_dev_std = np.zeros((len(anchor_dev), 34))
    anchor_dev_std[range(len(anchor_dev)), anchor_dev] = 1
    anchor_test_std = np.zeros((len(anchor_test), 34))
    anchor_test_std[range(len(anchor_test)), anchor_test] = 1
    anchor_test1_std = np.zeros((len(anchor_test1), 34))
    anchor_test1_std[range(len(anchor_test1)), anchor_test1] = 1
    '''
    anchor_test2_std = np.zeros((len(anchor_test2), 34))
    anchor_test2_std[range(len(anchor_test2)), anchor_test2] = 1
    anchor_test3_std = np.zeros((len(anchor_test3), 34))
    anchor_test3_std[range(len(anchor_test3)), anchor_test3] = 1
    '''
    print("demension: %d, train_size: %d, test_size: %d"
          %( 300, sent_train.shape[0], sent_dev.shape[0]))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cf = config()
            cnn = ed_model(cf, vocab_length, vectors )
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        final_prefix = os.path.join(checkpoint_dir, "final")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()


        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, size_batch):
            """
            A single training step
            """
            global e
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5,
                cnn.size_batch : size_batch
            }
            _, step, summaries, loss = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("\rtraining:{}:epoch {} step {}, loss {:g}".format(time_str,e, step, loss))
            train_summary_writer.add_summary(summaries, step)
        final = []
        def dev_step(x_batch, y_batch):
            """
            A single training step
            """
            global e
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5,
                cnn.size_batch : len(x_batch)
            }
            _, step, summaries, loss = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("\rdevolped: {}:epoch {} step {}, loss {:g}".format(time_str,e, step, loss))
            train_summary_writer.add_summary(summaries, step)
        
        def test_step(x, y_batch, y, writer=None):
            """
            Evaluates model on a dev set
            """
            global final
            feed_dict = {
                cnn.input_x: x,
                cnn.input_y : y,
                cnn.dropout_keep_prob: 0.5,
                cnn.size_batch : len(x)
            }
            step, summaries, y_pred = sess.run(
                [global_step, dev_summary_op, cnn.predictions],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            precision, recall, f1_score, status = precision_recall_fscore_support(y_batch, np.array(y_pred), 
                                                                                  labels=range(1,34),
                                                                                  pos_label=None,
                                                                                  average='micro')
            print("{}: step {}:".format(time_str, step))
            print(precision, recall, f1_score)
            if writer:
                writer.add_summary(summaries, step)
            final.append((precision, recall, f1_score))
            if f1_score >= 0.63:
                return True
            else: return False

        # Generate batches
        stop = False
        for e in np.arange(FLAGS.num_epochs):
            if stop == True:
                break
            for step, (x_batch, y_batch) in enumerate(data_iterator(
                        sent_train, anchor_train_std, cf.batch_size)):
                # Training loop. For each batch...

                size_batch = len(x_batch)
                train_step(x_batch, y_batch, size_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nDevolope:")
                    for step, (x_d, y_d) in enumerate(data_iterator(
                            sent_dev, anchor_dev_std, cf.batch_size)):
                        dev_step(x_d, y_d)
                    print("")
                    print("Evaluate:")
                    stop = test_step(sent_test, anchor_test, anchor_test_std)
                    if stop == True:
                        break
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
        pickle.dump(final, open("final.bin", "wb"))
        print("Evaluate:")
        print("Training:")
        test_step(sent_test, anchor_test, anchor_test_std)
        print("Test case 1:")
        test_step(sents_test1, anchor_test1, anchor_test1_std)
        print("Test case 2:")
        '''
        test_step(sents_test2, anchor_test2, anchor_test2_std)
        print("Test case 3:")
        test_step(sents_test3, anchor_test3, anchor_test3_std)
        '''
        print("")
        path = saver.save(sess, final_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))
