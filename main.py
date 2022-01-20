"""Code for training PMKL."""
from datetime import datetime
import json
import numpy as np
import os
import random
from scipy.misc import imsave
import matplotlib.pyplot as plt
import time

import tensorflow as tf

import data_loader, losses
import model as model
from utils import *

import csv
from stats_func import *

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow

import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
restore_checkpoint = True
restore_epoch = 0
save_all_interval = 25
restore_path = ''
restore_path_teacher = ''

train_list_pth = ''
valid_list_pth = ''
train_data_pth = ''
evaluation_interval = 10
visual_interval = 300
save_interval = 500
drop_rate_value=0.0
is_training_value=True
output_root_dir = './output'
modality_list = [
    'Flair',
    'T1c',
    'T1',
    'T2',
]

kd_weight = 0.5
contras_weight = 0.5
BATCH_SIZE = 4
CROP_SIZE = 80
NUM_CHANNEL = 4
NUM_CLS = 5
TEMP = 10.0
max_inter = 300
max_epoch = 200
BASE_LEARNING_RATE = 0.001
save_num_images = 1

class MultiModal:

    def __init__(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.drop_rate=tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=())
        self._output_dir = os.path.join(output_root_dir, current_time)

    def model_setup(self):

        self.input_flair = tf.placeholder(tf.float32, [BATCH_SIZE,CROP_SIZE,CROP_SIZE,CROP_SIZE,1,], name="input_flair")
        self.input_t1c = tf.placeholder(tf.float32, [BATCH_SIZE,CROP_SIZE,CROP_SIZE,CROP_SIZE,1,], name="input_t1c")
        self.input_t1 = tf.placeholder(tf.float32, [BATCH_SIZE,CROP_SIZE,CROP_SIZE,CROP_SIZE,1,], name="input_t1")
        self.input_t2 = tf.placeholder(tf.float32, [BATCH_SIZE,CROP_SIZE,CROP_SIZE,CROP_SIZE,1,], name="input_t2")

        self.input_brainmask = tf.placeholder(tf.float32, [BATCH_SIZE,CROP_SIZE,CROP_SIZE,CROP_SIZE,NUM_CHANNEL,], name="input_brainmask")
        self.input_label = tf.placeholder(tf.float32, [BATCH_SIZE,CROP_SIZE,CROP_SIZE,CROP_SIZE,NUM_CLS,], name="label")
        self.input_prs = tf.placeholder(tf.float32, [BATCH_SIZE,CROP_SIZE,CROP_SIZE,CROP_SIZE,NUM_CHANNEL,], name="input_prs")
        self.input_abs = tf.placeholder(tf.float32, [BATCH_SIZE,CROP_SIZE,CROP_SIZE,CROP_SIZE,4,], name="input_abs")

        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
        self.lr_summ = tf.summary.scalar("lr", self.learning_rate)

        inputs = {
            'input_prs': self.input_prs,
            'input_abs': self.input_abs,
            'input_brainmask': self.input_brainmask,
            'input_label': self.input_label,
            'input_flair': self.input_flair,
            'input_t1c': self.input_t1c,
            'input_t1': self.input_t1,
            'input_t2': self.input_t2,
        }

        outputs = model.get_outputs(inputs, NUM_CLS, is_training=self.is_training, drop_rate=self.drop_rate)

        self.feature_prs = outputs['feature_prs']
        self.feature_flair = outputs['feature_flair']
        self.feature_t1c__ = outputs['feature_t1c__']
        self.feature_t1___ = outputs['feature_t1___']
        self.feature_t2___ = outputs['feature_t2___']

        self.feat_prs = outputs['feat_prs']
        self.feat_abs = outputs['feat_abs']

        self.d1_out_prs = outputs['d1_out_prs']
        self.d1_out_abs = outputs['d1_out_abs']

        self.seg_logit_prs = outputs['seg_logit_prs']
        self.seg_pred_prs = outputs['seg_pred_prs']

        self.seg_logit_abs = outputs['seg_logit_abs']
        self.seg_pred_abs = outputs['seg_pred_abs']

        self.seg_pred_compact_prs = tf.argmax(self.seg_pred_prs, axis=4)
        self.seg_pred_compact_abs = tf.argmax(self.seg_pred_abs, axis=4)

        self.seg_pred_compact = tf.argmax((self.seg_pred_prs + self.seg_pred_abs) / 2.0, axis=4)
        self.input_label_compact = tf.argmax(tf.cast(self.input_label, tf.int64), axis=4)

        self.dice_arr_prs = dice_eval(self.seg_pred_compact_prs, self.input_label_compact, NUM_CLS)
        self.dice_c1_prs = self.dice_arr_prs[0]
        self.dice_c2_prs = self.dice_arr_prs[1]
        self.dice_c3_prs = self.dice_arr_prs[2]
        self.dice_c4_prs = self.dice_arr_prs[3]
        self.dice_avg_prs = (self.dice_c1_prs+self.dice_c2_prs+self.dice_c3_prs)/3.0

        self.dice_arr_abs = dice_eval(self.seg_pred_compact_abs, self.input_label_compact, NUM_CLS)
        self.dice_c1_abs = self.dice_arr_abs[0]
        self.dice_c2_abs = self.dice_arr_abs[1]
        self.dice_c3_abs = self.dice_arr_abs[2]
        self.dice_c4_abs = self.dice_arr_abs[3]
        self.dice_avg_abs = (self.dice_c1_abs + self.dice_c2_abs + self.dice_c3_abs) / 3.0

        self.dice_arr = dice_eval(self.seg_pred_compact, self.input_label_compact, NUM_CLS)
        self.dice_c1 = self.dice_arr[0]
        self.dice_c2 = self.dice_arr[1]
        self.dice_c3 = self.dice_arr[2]
        self.dice_c4 = self.dice_arr[3]
        self.dice_avg = (self.dice_c1 + self.dice_c2 + self.dice_c3) / 3.0

        self.dice_avg_prs_summ = tf.summary.scalar('dice_avg_prs', self.dice_avg_prs)
        self.dice_avg_abs_summ = tf.summary.scalar('dice_avg_abs', self.dice_avg_abs)
        self.dice_avg_summ = tf.summary.scalar('dice_avg', self.dice_avg)
        self.dice_merge_summ = tf.summary.merge(
            [self.dice_avg_prs_summ, self.dice_avg_abs_summ,
             self.dice_avg_summ])

        # Image visualization
        images_summary = tf.py_func(decode_images, [self.input_prs, save_num_images], tf.uint8)
        labels_summary = tf.py_func(decode_labels, [self.input_label_compact, save_num_images, NUM_CLS], tf.uint8)
        preds_summary = tf.py_func(decode_labels, [self.seg_pred_compact_prs, save_num_images, NUM_CLS], tf.uint8)
        teacher_summary = tf.py_func(decode_labels,[self.seg_pred_compact_abs, save_num_images, NUM_CLS], tf.uint8)
        self.visual_summary = tf.summary.image('images',
                                               tf.concat(axis=2,
                                                         values=[images_summary, labels_summary, preds_summary, teacher_summary]),
                                               max_outputs=save_num_images)  # Concatenate row-wise.

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.
        """

        self.model_vars = tf.trainable_variables()
        self.var_prs = [v for v in self.model_vars if 'prs' in v.name or 'metric' in v.name]
        self.var_abs = [v for v in self.model_vars if 'abs' in v.name and not 'metric' in v.name]

        print 'var_prs'
        for v in self.var_prs:
            print v.name

        print 'var_abs'
        for v in self.var_abs:
            print v.name

        self.loss_prs = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.seg_logit_prs)
        self.loss_abs = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.seg_logit_abs)
        self.kd_mask_1 = tf.nn.relu(self.loss_prs - self.loss_abs)
        self.kd_mask_2 = tf.cast(tf.equal(self.seg_pred_compact_abs, self.input_label_compact), tf.float32)
        self.kd_mask = self.kd_mask_1 * self.kd_mask_2
        self.kd_mask_nograd = tf.stop_gradient(self.kd_mask)


        self.kd_ce_loss_pixel = losses.kd_loss(student=self.seg_logit_prs, teacher=self.seg_logit_abs,
                                               hard_label=self.input_label, num_cls=NUM_CLS, temperature=TEMP)

        self.kd_ce_loss = tf.reduce_sum(self.kd_ce_loss_pixel * self.kd_mask_nograd)/tf.reduce_sum(self.kd_mask_nograd)

        self.ce_loss_prs, self.dice_loss_prs = losses.task_loss(self.seg_pred_prs, self.input_label, NUM_CLS)
        self.ce_loss_abs, self.dice_loss_abs = losses.task_loss(self.seg_pred_abs, self.input_label, NUM_CLS)

        self.l2_loss_prs = tf.add_n([0.00001 * tf.nn.l2_loss(v) for v in self.var_prs if '/kernel' in v.name])
        self.l2_loss_abs = tf.add_n([0.00001 * tf.nn.l2_loss(v) for v in self.var_abs if '/kernel' in v.name])


        for ii in range(BATCH_SIZE):
            for jj in range(BATCH_SIZE):
                if jj!=ii:
                    if ii==0 and jj==1:
                        self.embed_anchor = tf.concat((self.feat_prs, tf.reshape(self.feat_prs[ii],[1,-1])), axis=0)
                        self.embed_pair = tf.concat((self.feat_abs, tf.reshape(self.feat_abs[jj],[1,-1])), axis=0)
                    else:
                        self.embed_anchor = tf.concat((self.embed_anchor, tf.reshape(self.feat_prs[ii],[1,-1])), axis=0)
                        self.embed_pair = tf.concat((self.embed_pair, tf.reshape(self.feat_abs[jj],[1,-1])), axis=0)

        contras_label = np.ones(BATCH_SIZE)
        contras_label = tf.concat((contras_label, np.zeros((BATCH_SIZE-1)*BATCH_SIZE)), axis=0)

        self.contras_loss = tf.contrib.losses.metric_learning.contrastive_loss(contras_label, self.embed_anchor, self.embed_pair, margin=1.0)

        seg_loss_prs = self.l2_loss_prs + (self.dice_loss_prs + self.ce_loss_prs) + kd_weight * self.kd_ce_loss + contras_weight * self.contras_loss

        optimizer_prs = tf.train.MomentumOptimizer(self.learning_rate, 0.9)

        self.s_trainer_prs = optimizer_prs.minimize(seg_loss_prs, var_list=self.var_prs)

        # Summary variables for tensorboard
        self.ce_loss_prs_summ = tf.summary.scalar("ce_loss_prs", self.ce_loss_prs)
        self.ce_loss_abs_summ = tf.summary.scalar("ce_loss_abs", self.ce_loss_abs)
        self.dice_loss_prs_summ = tf.summary.scalar("dice_loss_prs", self.dice_loss_prs)
        self.dice_loss_abs_summ = tf.summary.scalar("dice_loss_abs", self.dice_loss_abs)
        self.l2_loss_prs_summ = tf.summary.scalar("l2_loss_prs", self.l2_loss_prs)
        self.l2_loss_abs_summ = tf.summary.scalar("l2_loss_abs", self.l2_loss_abs)
        self.kd_ce_loss_summ = tf.summary.scalar("kd_ce_loss", self.kd_ce_loss)
        self.contras_loss_summ = tf.summary.scalar("contras_loss", self.contras_loss)

        self.s_loss_merge_prs_summ = tf.summary.merge([self.ce_loss_prs_summ, self.dice_loss_prs_summ, self.l2_loss_prs_summ, self.kd_ce_loss_summ, self.contras_loss_summ])
        self.s_loss_merge_abs_summ = tf.summary.merge([self.ce_loss_abs_summ, self.dice_loss_abs_summ, self.l2_loss_abs_summ])

    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder

        data, brainmask, label = data_loader.load_data(train_list_pth, train_data_pth, modality_list, BATCH_SIZE, gt_flag=True, crop_size=CROP_SIZE, num_cls=NUM_CLS)

        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())

        restore_var_list_teacher = {}
        for v in self.var_abs:
            restore_var_list_teacher.update({v.name[:-2].replace('_abs',''): v})

        warming_var_list = {}
        for v in tf.global_variables():
            if 'prs' in v.name and 'metric' not in v.name:
                warming_var_list.update({v.name[:-2]: v})

        restorer_teacher = tf.train.Saver(var_list=restore_var_list_teacher)
        saver = tf.train.Saver(var_list=self.model_vars, max_to_keep=10000)
        saver_all_variable = tf.train.Saver(max_to_keep=10000)
        warmer = tf.train.Saver(var_list=warming_var_list)

        with open(train_list_pth, 'r') as fp:
            rows = fp.readlines()

        max_images = len(rows)
        time_st = time.time()

        if 'session' in locals() and session is not None:
            print('Close interactive session')
            session.close()

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            if restore_checkpoint:
                if restore_epoch > 0:
                    warmer.restore(sess, restore_path)
                else:
                    restorer_teacher.restore(sess, restore_path_teacher)

            writer = tf.summary.FileWriter(self._output_dir)
            writer_val = tf.summary.FileWriter(self._output_dir+'/val')

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            curr_lr = BASE_LEARNING_RATE
            for epoch in range(restore_epoch, max_epoch):
                print("In the epoch ", epoch)

                curr_lr = BASE_LEARNING_RATE*(1.0-np.float32(epoch)/np.float32(max_epoch))**(0.9)

                for i in range(0, max_inter):
                    starttime = time.time()
                    data_feed, label_feed = sess.run([data, label])

                    data_m1 = data_feed[:, :, :, :CROP_SIZE, 0:1].copy()
                    data_m2 = data_feed[:, :, :, CROP_SIZE:2 * CROP_SIZE, 0:1].copy()
                    data_m3 = data_feed[:, :, :, 2 * CROP_SIZE:3 * CROP_SIZE, 0:1].copy()
                    data_m4 = data_feed[:, :, :, 3 * CROP_SIZE:, 0:1].copy()
                    input_prs_feed = np.concatenate((data_m2, data_m2, data_m2, data_m2), axis=-1)
                    input_abs_feed = np.concatenate((data_m1, data_m2, data_m3, data_m4), axis=-1)

                    _, summary_str, dice_loss_prs_val, kd_ce_loss_val, contras_loss_val = sess.run(
                        [self.s_trainer_prs, self.s_loss_merge_prs_summ, self.dice_loss_prs, self.kd_ce_loss, self.contras_loss],
                        feed_dict={
                            self.input_prs: input_prs_feed,
                            self.input_abs: input_abs_feed,
                            self.input_label: label_feed,
                            self.input_flair: data_m1,
                            self.input_t1c: data_m2,
                            self.input_t1: data_m3,
                            self.input_t2: data_m4,
                            self.learning_rate: curr_lr,
                            self.drop_rate: drop_rate_value,
                            self.is_training: is_training_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    summary_str = sess.run(self.lr_summ,
                             feed_dict={
                                 self.learning_rate: curr_lr,
                             })
                    writer.add_summary(summary_str, epoch * max_inter + i)
                    writer.flush()

                    if (i+1) % evaluation_interval==0:
                        summary_str = sess.run(
                            self.dice_merge_summ, feed_dict={
                            self.input_prs: input_prs_feed,
                            self.input_abs: input_abs_feed,
                            self.input_label: label_feed,
                            self.input_flair: data_m1,
                            self.input_t1c: data_m2,
                            self.input_t1: data_m3,
                            self.input_t2: data_m4,
                            self.is_training: False,
                            self.drop_rate: 0.0,
                        })
                        writer.add_summary(summary_str, epoch * max_inter + i)
                        writer.flush()

                    if (i + 1) % visual_interval == 0:
                        summary_img = sess.run(self.visual_summary, feed_dict={
                                self.input_prs: input_prs_feed,
                                self.input_abs: input_abs_feed,
                                self.input_label: label_feed,
                                self.input_flair: data_m1,
                                self.input_t1c: data_m2,
                                self.input_t1: data_m3,
                                self.input_t2: data_m4,
                                self.is_training: False,
                                self.drop_rate: 0.0,
                            })
                        writer.add_summary(summary_img, epoch * max_inter + i)
                        writer.flush()

                    if (epoch * max_inter + i + 1) % save_interval==0:
                        saver.save(sess, os.path.join(
                            self._output_dir, "multimodal"), global_step=epoch * max_inter + i)

                    print("Processed batch {}/{}  time {:.2f}, dice_loss {:.2f}, kd_ce_loss {:.4f}, contras_loss {:.4f}".format(
                                i, max_inter, time.time() - starttime, dice_loss_prs_val,
                                kd_weight * kd_ce_loss_val, contras_weight * contras_loss_val))

                if (epoch + 1) % save_all_interval == 0:
                    saver_all_variable.save(sess, os.path.join(
                        self._output_dir, "multimodal-all-variable"), global_step=epoch * max_inter + i)

            coord.request_stop()
            coord.join(threads)


def main():

    multimodal_model = MultiModal()
    multimodal_model.train()


if __name__ == '__main__':
    main()
