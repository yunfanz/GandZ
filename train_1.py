import tensorflow as tf
from resnet import RESNET, UPDATE_OPS_COLLECTION, RESNET_VARIABLES, MOVING_AVERAGE_DECAY
from dcm_reader import DCMReader
import numpy as np
import os
import time
MOMENTUM = 0.9
SP2_BOX = (210,180,210,1)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './tmp/resnet_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'DATA/train/',
                           """Directory where data is located""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 1, "batch size")
tf.app.flags.DEFINE_integer('num_per_epoch', None, "max steps per epoch")
tf.app.flags.DEFINE_integer('epoch', 1, "number of epochs to train")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('is_training', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')
def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size
    #return num_correct
def test(sess, net, is_training, validation=False):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    
    coord = tf.train.Coordinator()
    reader = load_dcm(coord, FLAGS.data_dir)
    corpus_size = reader.corpus_size
    #import IPython; IPython.embed()
    batch, labels = reader.dequeue(FLAGS.batch_size)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)


    logits = net.inference(batch)
    #loss_ = net.loss(logits, labels)
    predictions = tf.nn.softmax(logits)
    #import IPython; IPython.embed()
    #top1_error = top_k_error(predictions, labels, 1)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    #import IPython; IPython.embed()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print("No checkpoint to continue from in", FLAGS.train_dir)
            sys.exit(1)
        print("resume", latest)
        saver.restore(sess, latest)

    
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    try:
        if FLAGS.num_per_epoch:
            batch_idx = min(FLAGS.num_per_epoch, corpus_size) // FLAGS.batch_size
        else:
            batch_idx = corpus_size // FLAGS.batch_size
        for idx in range(batch_idx):
            start_time = time.time()
            step = sess.run(global_step)

            o = sess.run(i, { is_training: True })

            loss_value = o[1]

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 1 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('Epoch %d, [%d / %d], loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (epoch, idx, batch_idx, loss_value, examples_per_sec, duration))

            if write_summary:
                summary_str = o[2]
                summary_writer.add_summary(summary_str, step)

            _, top1_error_value = sess.run([val_op, top1_error], { is_training: False })
            #pp, ll = sess.run([predictions, labels], {is_training:False})
            #print('Predictions: ', pp)
            #print('labels: ', ll)
            print('Validation top1 error %.2f' % top1_error_value)

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
        #G
    finally:
        print('Finished, output see {}'.format(FLAGS.train_dir))
        coord.request_stop()
        coord.join(threads)

def train(sess, net, is_training):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    
    coord = tf.train.Coordinator()
    reader = load_dcm(coord, FLAGS.data_dir)
    corpus_size = reader.corpus_size
    #import IPython; IPython.embed()
    train_batch, labels = reader.dequeue(FLAGS.batch_size)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    if False:
        train_batch = tf.transpose(train_batch, [3,1,2,0])  #treat slice index as sub_batch index

    logits = net.inference(train_batch)

    #import IPython; IPython.embed()

    loss_ = net.loss(logits, labels)
    predictions = tf.nn.softmax(logits)
    #import IPython; IPython.embed()
    top1_error = top_k_error(predictions, labels, 1)


    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.scalar_summary('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.99, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.scalar_summary('val_top1_error_avg', top1_error_avg)

    tf.scalar_summary('learning_rate', FLAGS.learning_rate)
    ###
    #opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.6, beta2=0.999, epsilon=1e-5)
    ###
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    #import IPython; IPython.embed()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print("No checkpoint to continue from in", FLAGS.train_dir)
            sys.exit(1)
        print("resume", latest)
        saver.restore(sess, latest)

    
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    try:
        for epoch in range(FLAGS.epoch):
            if FLAGS.num_per_epoch:
                batch_idx = min(FLAGS.num_per_epoch, corpus_size) // FLAGS.batch_size
            else:
                batch_idx = corpus_size // FLAGS.batch_size
            for idx in range(batch_idx):
                start_time = time.time()

                step = sess.run(global_step)
                i = [train_op, loss_]

                write_summary = step % 100 and step > 1
                if write_summary:
                    i.append(summary_op)

                o = sess.run(i, { is_training: True })

                loss_value = o[1]

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 1 == 0:
                    examples_per_sec = FLAGS.batch_size / float(duration)
                    format_str = ('Epoch %d, [%d / %d], loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (epoch, idx, batch_idx, loss_value, examples_per_sec, duration))

                if write_summary:
                    summary_str = o[2]
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step > 1 and step % 100 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

                # Run validation periodically
                if step > 1 and step % 10 == 0:
                    _, top1_error_value = sess.run([val_op, top1_error], { is_training: False })
                    #pp, ll = sess.run([predictions, labels], {is_training:False})
                    #print('Predictions: ', pp)
                    #print('labels: ', ll)
                    print('Validation top1 error %.2f' % top1_error_value)

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
        #G
    finally:
        print('Finished, output see {}'.format(FLAGS.train_dir))
        coord.request_stop()
        coord.join(threads)

def load_dcm(coord, data_dir):
    if not data_dir:
        data_dir = './DATA/train/'
    print('Using data_dir', data_dir)

    reader = DCMReader(
        data_dir,
        coord,
        e2e=True,
        queue_size=32, 
        byPatient=True, 
        q_shape=SP2_BOX, 
        n_threads=1)
    return reader


def main(_):

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=sessconfig)
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    
    is_training = tf.placeholder('bool', [], name='is_training')
    net = RESNET(sess, 
                dim=3,
                num_classes=2,
                num_blocks=[3, 4, 3],  # first chan is not a block
                num_chans=[16,16,32,64],
                use_bias=False, # defaults to using batch norm
                bottleneck=True,
                is_training=True)
    
    train(sess, net, is_training)


if __name__ == '__main__':
    tf.app.run()
