from __future__ import print_function

import os
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np

from utils import *

from functools import partial
import matplotlib.pyplot as plt

def make_gaussian_2d_kernel(sigma, truncate=4.0, dtype=tf.float32):
    radius = tf.to_int32(sigma * truncate)
    x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
    k = tf.exp(-0.5 * tf.square(x / sigma))
    k = k / tf.reduce_sum(k) # 高斯核归一化
    return tf.expand_dims(k, 1) * k

def Gaussian_kernel(input):
    kernel = make_gaussian_2d_kernel(1)
    kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
    output = tf.nn.separable_conv2d(input, kernel, tf.eye(3, batch_shape=[1, 1]),strides=[1, 1, 1, 1], padding='SAME')
    return output

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def dilation2d(img4D):
    with tf.variable_scope('dilation2d'):
        kernel = tf.ones((3, 3, img4D.get_shape()[3])) 
        output4D = tf.nn.dilation2d(img4D, filter=kernel, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
        output4D = output4D - tf.ones_like(output4D)

        return output4D


def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, channel=64, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_im = concat([input_max, input_im])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")
        for idx in range(layer_num):
            conv = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=partial(tf.nn.leaky_relu, alpha = 0.01), name='activated_layer_%d' % idx)
        conv = tf.layers.conv2d(conv, 4, kernel_size, padding='same', activation=None, name='recon_layer')

    R = tf.sigmoid(conv[:,:,:,0:3])
    L = tf.sigmoid(conv[:,:,:,3:4])

    return R, L

def RASB(input_img4D, channel=64, kernel_size=3):
    conv1 = tf.layers.conv2d(input_img4D, channel, kernel_size, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, channel, kernel_size, padding='same', activation=None)
    img4D_attentioned = spatial_attention(conv2)
    output_img4D = input_img4D + img4D_attentioned
    return output_img4D

def spatial_attention(inputs_img4D):
    maxpool_spatial=tf.reduce_max(inputs_img4D,axis=3,keepdims=True)
    avgpool_spatial=tf.reduce_mean(inputs_img4D,axis=3,keepdims=True)
    max_avg_pool_spatial=tf.concat([maxpool_spatial,avgpool_spatial],axis=3)
    conv_layer=tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=7, padding="same", activation=None)
    spatial_attention=tf.nn.sigmoid(conv_layer)
    attentioned_feature=inputs_img4D * spatial_attention
    return attentioned_feature




def RelightNet(input_L, input_R_1, input_R_2, channel=64, kernel_size=3):
    input_im = concat([input_R_1, input_R_2, input_L])
    with tf.variable_scope('RelightNet'):
        conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
        conv0 = dilation2d(conv0)
        conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.compat.v1.nn.swish)
        conv1 = dilation2d(conv1)
        conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.compat.v1.nn.swish)
        conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(conv3, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)

        conv4 = RASB(conv4)
        up1 = tf.image.resize_nearest_neighbor(conv4, (tf.shape(conv3)[1], tf.shape(conv3)[2]))
        deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv3
        deconv1 = RASB(deconv1)
        up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
        deconv2= tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        deconv2 = RASB(deconv2)
        up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.compat.v1.nn.swish) + conv1
        deconv3 = RASB(deconv3)
        up4 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv4 = tf.layers.conv2d(up4, channel, kernel_size, padding='same', activation=tf.compat.v1.nn.swish) + conv0
        deconv4 = RASB(deconv4)
        
        deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv4)[1], tf.shape(deconv4)[2]))
        deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv4)[1], tf.shape(deconv4)[2]))
        deconv3_resize = tf.image.resize_nearest_neighbor(deconv3, (tf.shape(deconv4)[1], tf.shape(deconv4)[2]))        
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3_resize, deconv4])
        feature_fusion = tf.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        output = tf.layers.conv2d(feature_fusion, 1, 3, padding='same', activation=None)
    return output

class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.DecomNet_layer_num = 5

        # Ori
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)
        [R_high, I_high] = DecomNet(self.input_high, layer_num=self.DecomNet_layer_num)
        
        R_low_Relight_1 = Gaussian_kernel(R_low)
        Gaussian_noise = tf.keras.layers.GaussianNoise(0.2)
        R_low_Relight_2 = Gaussian_noise(R_low)
        I_delta = RelightNet(I_low, R_low_Relight_1, R_low_Relight_2) # concat

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])

        self.output_R_low = R_low
        self.output_I_low = I_low_3
        self.output_R_high = R_high
        self.output_I_high = I_high_3
        self.output_I_delta = I_delta_3
        self.output_S = R_low * I_delta_3

        # Sobel
        Sobel_low = tf.image.sobel_edges(R_low)
        Sobel_high = tf.image.sobel_edges(R_high)

        self.output_Sobel_low = Sobel_low
        self.output_Sobel_high = Sobel_high

        self.sobel_loss = tf.reduce_mean(tf.abs(Sobel_low - Sobel_high))

        Sobel_low_rgb = Sobel_low[:,:,:,:,0]+ Sobel_low[:,:,:,:,1]
        Sobel_high_rgb = Sobel_high[:,:,:,:,0]+ Sobel_high[:,:,:,:,1]
        self.sobel_ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(Sobel_low_rgb, Sobel_high_rgb, max_val = 1))

        # Log
        Log_input_high_1 = tf.math.log(255 * self.input_high)
        Log_input_high_2 = tf.where(tf.math.is_inf(Log_input_high_1), tf.ones_like(Log_input_high_1), Log_input_high_1)
        
        Log_input_low_1 = tf.math.log(255 * self.input_low)
        Log_input_low_2 = tf.where(tf.math.is_inf(Log_input_low_1), tf.ones_like(Log_input_low_1), Log_input_low_1)

        Log_I_high = tf.math.log(255 * I_high)
        Log_I_high_3 = concat([Log_I_high, Log_I_high, Log_I_high])
        Log_I_low = tf.math.log(255 * I_low)
        Log_I_low_3 = concat([Log_I_low, Log_I_low, Log_I_low])

        Log_R_low = tf.math.log(255 * R_low)
        Log_R_high = tf.math.log(255 * R_high)

        self.output_Log_input_high_1 = Log_input_high_1
        self.output_Log_input_high_2 = Log_input_high_2
        self.output_Log_I_high_3 = Log_I_high_3
        self.output_Log_R_low = Log_R_low
        
        self.Log_recon_loss_mutal_high = tf.reduce_mean(tf.abs(Log_R_low + Log_I_high_3 - Log_input_high_2))
        self.Log_recon_loss_mutal_low = tf.reduce_mean(tf.abs(Log_R_high + Log_I_low_3 - Log_input_low_2))
        self.Log_recon_loss_high = tf.reduce_mean(tf.abs(Log_R_high + Log_I_high_3 - Log_input_high_2))
        self.Log_equal_R_loss = tf.reduce_mean(tf.abs(Log_R_low - Log_R_high))


        # Ori loss
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  self.input_low))
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - self.input_high))
        self.recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - self.input_low))
        self.recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - self.input_high))
        self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))

        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        # Ori my loss

        # R smooth loss
        self.Rsmooth_loss_low = self.smooth_R(I_low, R_low)
        self.Rsmooth_loss_high = self.smooth_R(I_high, R_high)

        # ssim loss
        self.ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(R_low, R_high, max_val = 1))



        # Decom TOTAL loss
        self.loss_Decom_retinex = self.recon_loss_low + self.recon_loss_high + 0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + 0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high + 0.01 * self.equal_R_loss
        self.loss_Decom_ori = self.loss_Decom_retinex 
        # + 0.001 * self.ssim_loss + 0.0015 * self.Rsmooth_loss_low + 0.001 * self.Rsmooth_loss_high 
        
        self.loss_Decom_sobel = 0.007 * self.sobel_loss + 0.007 * self.sobel_ssim_loss
        
        self.loss_Decom_Log = 0.0001 * self.Log_recon_loss_mutal_high + 0.006 * self.Log_equal_R_loss + 0.0005 * self.Log_recon_loss_mutal_low
        
        self.loss_Decom = self.loss_Decom_ori 
        # + self.loss_Decom_sobel 
        # + self.loss_Decom_Log
        



        # Relight TOTAL loss
        self.relight_loss_1 = tf.reduce_mean(tf.abs(R_low_Relight_1 * I_delta_3 - self.input_high))
        self.relight_loss_2 = tf.reduce_mean(tf.abs(R_low_Relight_1 * I_delta_3 - self.input_high))
        self.relight_ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(R_low * I_delta_3, self.input_high, max_val = 1))
        self.relight_I_loss = 1 - tf.reduce_mean(tf.image.ssim(I_delta_3, I_high_3, max_val = 1))
        self.loss_Relight = 0.5 * self.relight_loss_1 + 0.5 * self.relight_loss_2 + 3 * self.Ismooth_loss_delta + 0.5 * self.relight_I_loss +  0.5 * self.relight_I_loss



        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.var_Relight = [var for var in tf.trainable_variables() if 'RelightNet' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
        self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list = self.var_Relight)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom)
        self.saver_Relight = tf.train.Saver(var_list = self.var_Relight)

        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))

    def smooth_R(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.ave_gradient(input_R, "x") * tf.exp(-10 * self.gradient(input_I, "x")) + self.ave_gradient(input_R, "y") * tf.exp(-10 * self.gradient(input_I, "y")))

    def evaluate(self, epoch_num, eval_low_data, eval_high_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        # low
        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2, result_3 = self.sess.run([self.output_R_low, self.output_I_low, self.output_Sobel_low], feed_dict={self.input_low: input_low_eval})
                result_sobel = (result_3[:,:,:,:,0]+result_3[:,:,:,:,1])
                save_images(os.path.join(sample_dir, 'eval_I_low_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1)
                save_images(os.path.join(sample_dir, 'eval_Sobel_low_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_sobel)
            if train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta], feed_dict={self.input_low: input_low_eval})
                save_images(os.path.join(sample_dir, 'eval_output_S_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1)
                save_images(os.path.join(sample_dir, 'eval_I_delta_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_2)

        for idx in range(len(eval_high_data)):# 遍历eval_high
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2, result_3 = self.sess.run([self.output_R_high, self.output_I_high, self.output_Sobel_high], feed_dict={self.input_high: input_high_eval})
                result_sobel = (result_3[:,:,:,:,0]+result_3[:,:,:,:,1])
                save_images(os.path.join(sample_dir, 'eval_R_high_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1)
                save_images(os.path.join(sample_dir, 'eval_Sobel_high_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_sobel)

    def train(self, train_low_data, train_high_data, eval_low_data, eval_high_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom
            saver = self.saver_Decom
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            saver = self.saver_Relight

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        loss_y = []
        loss_y_batch = []

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                _, loss = self.sess.run([train_op, train_loss], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, \
                                                                           self.lr: lr[epoch]})

                loss_y.append(loss)

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            loss_y_batch.append(np.mean(loss_y))
            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, eval_high_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)
        if train_phase == "Decom":        
            loss_x = range(0,len(loss_y_batch))
            plt.plot(loss_x, loss_y_batch, color="r",label= "all")
            plt.legend()
            plt.title("The average loss is %.6f" % np.mean(loss_y_batch))
            plt.savefig('Decom_model_loss.jpg')
            plt.show()
        print("[*] Finish training for phase %s." % train_phase)

        if train_phase == "Relight":        
            loss_x = range(0,len(loss_y_batch))
            plt.plot(loss_x, loss_y_batch, color="r",label= "all")
            plt.legend()
            plt.title("The average loss is %.6f" % np.mean(loss_y_batch))
            plt.savefig('Relight_model_loss.jpg')
            plt.show()
        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './model/Decom')
        load_model_status_Relight, _ = self.load(self.saver_Relight, './model/Relight')
        if load_model_status_Decom and load_model_status_Relight:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            [R_low, I_low, I_delta, S] = self.sess.run([self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S], feed_dict = {self.input_low: input_low_test})

            if decom_flag == 1:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            save_images(os.path.join(save_dir, name + "."   + suffix), S)

