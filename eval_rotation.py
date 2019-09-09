import tensorflow as tf
import os
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
import time
import sys
import importlib

os.environ['CUDA_VISIBLE_DEVICES']="3"
#tf.enable_eager_execution()
SAVE_DIR=''
flags = tf.app.flags
flags.DEFINE_string("videos_csv_path", SAVE_DIR, "Directory for config data")
flags.DEFINE_string("num_workers", 'UCF', "Num workers to train")
flags.DEFINE_integer("image_size", 112, "Image size (height, width) in pixels")
flags.DEFINE_integer('batch_size', 8, 'Batch size')
flags.DEFINE_string("model_def", 'nets.resnet3D', "Model definition [models.inception_resnet_v1, models.mobilenet]")
flags.DEFINE_string("model_iterations", '103000', "model_iterations")
flags.DEFINE_integer("lr_decay_epochs", 24000, "Number of epochs between learning rate decay")
flags.DEFINE_float("lr_decay_factor", 0.1, "Learning rate decay factor.")
flags.DEFINE_float("lr", 0.01, "Initial learning rate.")
flags.DEFINE_integer("iterations", 104000, "Number of iterations")
flags.DEFINE_string("logs_dir", SAVE_DIR+'/log', "Directory where to write event logs")
flags.DEFINE_string("checkpoint_dir", SAVE_DIR+'/checkpoints', "Directory where to write trained models and checkpoints")
FLAGS = flags.FLAGS

def image_augmentation(img):
    img = tf.random_crop(img,[112, 112, 3]) 
    return img

def random_flip_a_clip(imgs):
    uniform_random = random_ops.random_uniform([], 0, 1.0)
    mirror_cond = math_ops.less(uniform_random, .5)
    print('MIRROR_COND:')
    print(mirror_cond)
    result = control_flow_ops.cond(
             mirror_cond,
             lambda: tf.image.flip_left_right(imgs),
             lambda: imgs)
    return result
def get_input(images, labels, rot90, label90, rot180, label180, rot270, label270):
    
    images, labels, rot90, label90, rot180, label180, rot270, label270 = tf.train.batch(
          [images, labels, rot90, label90, rot180, label180, rot270, label270],
          batch_size=FLAGS.batch_size,
          num_threads=12,
          shapes=[(16, 112, 112, 3), (), (16, 112, 112, 3), (), (16, 112, 112, 3), (), (16, 112, 112, 3), ()],
          allow_smaller_final_batch=False,
          capacity=10 * FLAGS.batch_size,)
    labels = slim.one_hot_encoding(labels, 4)
    label90 = slim.one_hot_encoding(label90, 4)
    label180 = slim.one_hot_encoding(label180, 4)
    label270 =  slim.one_hot_encoding(label270, 4)
    return images, labels, rot90, label90, rot180, label180, rot270, label270

def main(unused_arg):
    tf.set_random_seed(15)

    network = importlib.import_module(FLAGS.model_def)
     
    videos_sub_path = os.path.join(FLAGS.videos_csv_path, FLAGS.num_workers, "video_csv_all.csv")
    videos_sub_path = np.loadtxt(videos_sub_path, dtype=np.str)
    
    num_videos = videos_sub_path.shape[0]
    print(num_videos)
    videos_sub_path = tf.convert_to_tensor(videos_sub_path)

    video_index = tf.train.slice_input_producer([tf.range(0, num_videos, delta=1, dtype=tf.int32)], shuffle=False, capacity=1000)    

    video_path = tf.string_join([SAVE_DIR, videos_sub_path[video_index[0]]], separator="/") + ".npy"
    video_contents = tf.read_file(video_path)
    video_contents = tf.decode_raw(video_contents, out_type=tf.uint8)[128:]
    num_frames = tf.shape(video_contents)[0] / (FLAGS.image_size * FLAGS.image_size * 3)
    video_contents = tf.reshape(video_contents, [num_frames, FLAGS.image_size, FLAGS.image_size, 3])

    start_id = 0
    end_id = num_frames-15
   
    def get_clip():
        anchor_range = tf.range(start_id, end_id, delta=1)
        anchor_range = tf.random_shuffle(anchor_range)
        num_anchors = 1
        anchor=anchor_range[:num_anchors]
        range_right = tf.range(16)
        imgs_id = anchor+range_right
        images = tf.gather(video_contents, imgs_id, axis=0)
        images = random_flip_a_clip(images) 
        
        #images = tf.image.resize_images(images,[112, 112])
        #images = tf.random_crop(img,[16, 112, 112, 3], seed=14)
        #images = tf.map_fn(image_augmentation, images) 
        images = tf.cast(images, tf.float32)
        rot90_images = tf.contrib.image.rotate(images, 90)
        rot180_images = tf.contrib.image.rotate(images, 180)
        rot270_images = tf.contrib.image.rotate(images, 270)
        return images, rot90_images, rot180_images, rot270_images, tf.constant(0), tf.constant(1), tf.constant(2), tf.constant(3)
   
    images, rot90_images, rot180_images, rot270_images, labels, labels90, labels180, labels270 = get_clip() 
    images, labels, rot90_images, labels90, rot180_images, labels180, rot270_images, labels270 = get_input(images, labels, rot90_images, labels90, rot180_images, labels180, rot270_images, labels270)
    input_images = tf.concat([images, rot270_images,rot90_images, rot180_images], axis=0)
    input_labels = tf.concat([labels, labels270, labels90, labels180], axis = 0)
    if FLAGS.model_def == 'nets.C3D':
        _, logits = network.C3D(input_images, 4)
    else:
        _, logits = network.R2Plus1DNet(input_images, num_classes=4, training=False)
    predictions = tf.argmax(logits, 1)
    gt = tf.argmax(input_labels, 1)
    acc = tf.metrics.accuracy(predictions=predictions, labels=gt)
    tf.summary.scalar("Accuracy", acc[1])
    
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=logits) 
    total_loss = tf.losses.get_total_loss(name='total_loss') 
  
    
    slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
    summary_op = tf.summary.merge_all()

    hooks = [tf.train.SummarySaverHook(output_dir=FLAGS.checkpoint_dir, summary_op=summary_op, save_secs=50)]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #v2r = tf.trainable_variables() 
    
    v2r =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="R2Plus1DNet")
    print(v2r)
    restorer = tf.train.Saver(v2r)
    print('v2r:%d'%len(v2r))
    with tf.train.MonitoredTrainingSession(config=config) as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        start_time = time.time()
        accuracy = 0.0
        for i in range(1665):
            
            ac, err, = sess.run([acc, total_loss,])
            accuracy = accuracy+ac[1]
            if i % 100 == 0:
              print('[%d/%d]\tTime %s\tLoss %s\tAcc %2.3f' %
                      (i, 1665, time.strftime('%Y-%m-%d %X', time.localtime()), err,ac[1]))
              sys.stdout.flush()
        print("ACCURACY: %2.3f" % accuracy)
        sys.stdout.flush()
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()

