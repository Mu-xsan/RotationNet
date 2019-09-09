import tensorflow as tf
import numpy as np
import time
import data_processing_classification
from data_processing_classification import convert_images_to_clip
import os
import os.path
from os.path import join
import logging
import time
import sys
import random
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import horovod.tensorflow as hvd
import random
import importlib
VERSION_NAME = ""

flags = tf.app.flags
flags.DEFINE_string("model_def", 'nets.r21d', "Model definition [nets.resnet3D, nets.C3D]")
flags.DEFINE_string("checkpoint_save_path", 'check_point_ar/{}'.format(VERSION_NAME), 'checkpoint save path')
flags.DEFINE_string("logs_save_path", 'log_ar/{}'.format(VERSION_NAME), 'logs save path')
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_integer('batch_size', 8, "training batch_size")
flags.DEFINE_integer('epochs', 300, 'training epochs num')
flags.DEFINE_string("train_list_path", 'lists/train1.list', 'training list path')
flags.DEFINE_integer("clip_length", 16, 'clip length for sampling')
flags.DEFINE_integer("crop_size", 112, 'crop size of frames')
flags.DEFINE_integer('num_classes', 101, 'number of classes')
flags.DEFINE_float('lr_decay_epoch', 100000, 'learning rate decay')
flags.DEFINE_float('lr_decay_factor', 0.998, 'leraning rate decay factor')
flags.DEFINE_string("checkpoint_exclude_scopes", 'dense', "Comma-separated list of scopes of variables to exclude when restoring from a checkpoint")
FLAGS = flags.FLAGS

def _get_init_fn():
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    allv = tf.all_variables()
    print('allv:%d'%len(allv))
    for var in allv:
        flag = False
        for exclusion in exclusions:
            if not (var.op.name.find(exclusion)==-1):
            #if var.op.name.startswith(exclusion):
                flag = True
        if not flag:
            variables_to_restore.append(var)
    print('v2r:%d'%len(variables_to_restore))
    return variables_to_restore

def get_dataset_size():
    lines = open(FLAGS.train_list_path, 'r')
    lines = list(lines)
    dataset_size = len(lines)
    return dataset_size

dataset_size = get_dataset_size()
total_iterations = (dataset_size // FLAGS.batch_size) * FLAGS.epochs

def main(_):
    network = importlib.import_module(FLAGS.model_def)
    def read_data(input_queue, clip_length=FLAGS.clip_length):
        label = input_queue[1]
        filename = input_queue[0]
        filename = filename + '.npy'
        file_contents = tf.read_file(filename)
        file_contents = tf.decode_raw(file_contents, out_type=tf.float32)[32:]
        num_frames = tf.shape(file_contents)[0] / (FLAGS.crop_size * FLAGS.crop_size * 3)
        clip = tf.reshape(file_contents, [num_frames, FLAGS.crop_size, FLAGS.crop_size, 3])
        clip = tf.image.resize_images(clip, [128, 171])
        clip = tf.random_crop(clip, [num_frames, FLAGS.crop_size, FLAGS.crop_size, 3])
        begin = tf.random_uniform([], minval=0,maxval=num_frames - clip_length,dtype=np.int32)
        clip = tf.slice(clip, [begin, 0, 0, 0], [clip_length, -1, -1, -1])
        return clip, label

    def train_data_loader():
        lines = open(FLAGS.train_list_path, 'r')
        lines = list(lines)
        lines = [line.strip('\n').split() for line in lines]
        random.shuffle(lines)
        clips = [line[0] for line in lines]
        labels = [int(line[1]) for line in lines]
        clips = tf.convert_to_tensor(clips, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        input_queue = tf.train.slice_input_producer([clips, labels], shuffle=False)
        clip, label = read_data(input_queue)
        clip_batch, label_batch = tf.train.shuffle_batch([clip, label], batch_size=FLAGS.batch_size, 
            num_threads=8, seed=19,
            shapes=[(FLAGS.clip_length, FLAGS.crop_size, FLAGS.crop_size, 3), ()],
            capacity=25*FLAGS.batch_size, min_after_dequeue=5*FLAGS.batch_size
        )
        return clip_batch, label_batch
    
    hvd.init()
    tf.set_random_seed(15 + hvd.rank())
    batch_clips, batch_labels = train_data_loader()
    ori_labels = batch_labels
    batch_labels = tf.one_hot(batch_labels, FLAGS.num_classes)
    
    tf.summary.image('batch_example', batch_clips[:, 0, :, :, :])
    batch_clips = batch_clips / 127.5 - 1
    
    feature, logits = network.R2Plus1DNet(batch_clips, True, FLAGS.num_classes, weight_decay=5e-4)
    
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels))
        tf.summary.scalar('entropy_loss', loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(batch_labels, 1)), np.float32))
        tf.summary.scalar('accuracy', accuracy)

    
    sys.stdout.flush()

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step,
                                               FLAGS.lr_decay_epoch, FLAGS.lr_decay_factor, staircase=True)
    #opt = tf.train.GradientDescentOptimizer(learning_rate * hvd.size())
    opt = tf.train.MomentumOptimizer(learning_rate * hvd.size(), 0.9)
    opt = hvd.DistributedOptimizer(opt)
    train_op = slim.learning.create_train_op(loss, opt, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([updates], train_op)
    
    summary_op = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if hvd.rank() == 0:
        hooks = [hvd.BroadcastGlobalVariablesHook(0),
                 tf.train.StopAtStepHook(last_step=total_iterations),
                 tf.train.SummarySaverHook(output_dir=FLAGS.logs_save_path, summary_op=summary_op, save_steps=10)]
    else:
        hooks = [hvd.BroadcastGlobalVariablesHook(0),
                 tf.train.StopAtStepHook(last_step=total_iterations)]

    config.gpu_options.visible_device_list = str(hvd.local_rank())

    checkpoint_dir = FLAGS.checkpoint_save_path if hvd.rank() == 0 else None
    v2r = _get_init_fn()
    restorer = tf.train.Saver(v2r)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks, config=config,
                                           save_checkpoint_steps=1000) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        #restorer.restore(sess, './checkpoints_rotation/ar_r21_2fc_r21d/model.ckpt-178800')
        while not sess.should_stop():
            step = tf.train.global_step(sess, global_step)
            start = time.time()
            _, loss_out, accuracy_out, labels_outs = sess.run([train_op, loss, accuracy, ori_labels])
            if step % 10 == 0:
                print(labels_outs)
                epoch_num = step // (dataset_size // FLAGS.batch_size)
                batch_num = step % (dataset_size // FLAGS.batch_size)
                print('Rank[%d], Epoch [%d], Batch [%d]: Loss is [%.5f]; Accuracy is [%.5f]'%(hvd.rank(), epoch_num, batch_num, loss_out, accuracy_out))
                sys.stdout.flush()
        coord.request_stop()
        coord.join(threads)


if __name__=="__main__":
    tf.app.run()
