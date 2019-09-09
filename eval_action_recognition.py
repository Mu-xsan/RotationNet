#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
import os.path
from os.path import join
import logging
import sys
import random
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import itertools
import math
import importlib
os.environ['CUDA_VISIBLE_DEVICES']="2"
VERSION_NAME = "ar_r21_eval"

flags = tf.app.flags

flags.DEFINE_string("model_def", 'nets.r21d', "Model definition [models.inception_resnet_v1, models.mobilenet]")
flags.DEFINE_string("checkpoint_save_path", 'checkpoints_ar/{}'.format(VERSION_NAME), 'checkpoint save path')
flags.DEFINE_string("logs_save_path", 'log_ar_eval/{}'.format(VERSION_NAME), 'logs save path')
flags.DEFINE_integer('batch_size', 8, "training batch_size")
flags.DEFINE_integer('epochs', 150, 'training epochs num')
flags.DEFINE_string("test_list_path", 'lists/test1.list', 'training list path')
flags.DEFINE_integer("clip_length", 16, 'clip length for sampling')
flags.DEFINE_integer("crop_size", 112, 'crop size of frames')
flags.DEFINE_integer('num_classes', 101, 'number of classes')
flags.DEFINE_float('lr_decay_epoch', 1000, 'learning rate decay')
flags.DEFINE_float('lr_decay_factor', 0.8, 'leraning rate decay factor')
flags.DEFINE_string('models_dir','checkpoints_test', 'testing checkpoint path')

flags.DEFINE_integer("save_summaries_secs", 600, "The frequency with which summaries are saved, in seconds")
flags.DEFINE_integer("save_interval_secs", 600, "The frequency with which the model is saved, in seconds")
FLAGS = flags.FLAGS

slim = tf.contrib.slim

def get_dataset_size():
    lines = open(FLAGS.test_list_path, 'r')
    lines = list(lines)
    dataset_size = len(lines)
    return dataset_size

def main(_):
    network = importlib.import_module(FLAGS.model_def)
    def read_data_10_clips(input_queue, clip_length=FLAGS.clip_length):
        label = input_queue[1]
        filename = input_queue[0]
        filename = filename + '.npy'
        file_contents = tf.read_file(filename)
        file_contents = tf.decode_raw(file_contents, out_type=tf.float32)[32:]
        num_frames = tf.shape(file_contents)[0] / (FLAGS.crop_size * FLAGS.crop_size * 3)
        clip = tf.reshape(file_contents, [num_frames, FLAGS.crop_size, FLAGS.crop_size, 3])
        clip = tf.image.resize_images(clip, [128, 171])
        clip = tf.image.crop_to_bounding_box(clip, 8, 30, 112, 112)
        each_start = (num_frames - clip_length) // 10
        clips = []
        for i in range(10):
            begin = i * each_start
            clips.append(tf.slice(clip, [begin, 0, 0, 0], [clip_length, -1, -1, -1]))
        clips = tf.stack(clips)           
        return clips, label

    def test_data_loader():
        lines = open(FLAGS.test_list_path, 'r')
        lines = list(lines)
        lines = [line.strip('\n').split() for line in lines]
        clips = [line[0] for line in lines]
        labels = [int(line[1]) for line in lines]
        clips = tf.convert_to_tensor(clips, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        input_queue = tf.train.slice_input_producer([clips, labels], shuffle=True)
        clip, label = read_data_10_clips(input_queue)
        clip_batch, label_batch = tf.train.batch([clip, label], batch_size=FLAGS.batch_size, 
            num_threads=12,
            shapes=[(10, FLAGS.clip_length, FLAGS.crop_size, FLAGS.crop_size, 3), ()],
            capacity=25*FLAGS.batch_size,
            allow_smaller_final_batch=False,
        )
        return clip_batch, label_batch
    
    batch_clips, batch_labels = test_data_loader()
    ori_labels = batch_labels
    batch_labels = tf.one_hot(batch_labels, FLAGS.num_classes)
    batch_clips = tf.reshape(batch_clips, [FLAGS.batch_size * 10, FLAGS.clip_length, FLAGS.crop_size, FLAGS.crop_size, 3])

    batch_clips = batch_clips / 127.5 - 1

    feature, logits = network.R2Plus1DNet(batch_clips, layer_sizes = [1,1,1,1], training = False, num_classes = FLAGS.num_classes, weight_decay=5e-4)


    with tf.name_scope('accuracy'):
        logits = tf.reshape(logits, [FLAGS.batch_size, 10, -1])
        logits = tf.reduce_sum(logits, axis=1)
        predictions = tf.argmax(logits, 1)
        gt = ori_labels
        accuracy = tf.metrics.accuracy(predictions=predictions, labels=gt)
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels))
        tf.summary.scalar('entropy_loss', loss)


    num_examples = get_dataset_size()
    batch_size = FLAGS.batch_size
    num_batches = math.ceil(num_examples / float(batch_size))
    num_batches = int(num_batches)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #v2r =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="R2Plus1DNet")
    v2r = tf.all_variables()
    print(v2r)
    restorer = tf.train.Saver(v2r)
    print('v2r:%d'%len(v2r))
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        #restorer.restore(sess, FLAGS.checkpoint_save_path + '/model.ckpt-39000')
        #print('restore from {}' % FLAGS.checkpoint_save_path + '/model.ckpt-100000')
        acc = 0
        for i in range(num_batches):
            ac, ls, lb, pred = sess.run([accuracy, loss, ori_labels, predictions])
            print(ac)
            print(lb)
            print(pred)
            acc = acc + ac[1]
            if i % 10 == 0:
              print('[%d/%d]\tTime %s\tLoss %s\tAcc %2.3f' %
                      (i, num_batches, time.strftime('%Y-%m-%d %X', time.localtime()), ls, acc / i))
              sys.stdout.flush()
        print("ACCURACY: %2.3f" % acc / num_batches)
        sys.stdout.flush()
        coord.request_stop()
        coord.join(threads)

        
if __name__ == '__main__':
    tf.app.run()
