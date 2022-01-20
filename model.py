"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf
import layers
import numpy as np

n_base_filters = 16
n_base_ch_se = 32
mlp_ch = 128
img_ch = 1
scale = 4
feat_dim=16


def get_outputs(inputs, num_cls, is_training=True, drop_rate=0.3):

        input_prs = inputs['input_prs']

        input_flair = inputs['input_flair']
        input_t1c__ = inputs['input_t1c']
        input_t1___ = inputs['input_t1']
        input_t2___ = inputs['input_t2']

        with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:
            feature_prs = feature_encoder(input_prs, is_training=is_training, drop_rate=drop_rate, name='ce_prs')

            feature_flair = feature_encoder(input_flair, is_training=is_training, drop_rate=drop_rate, name='ce_flair_abs')
            feature_t1c__ = feature_encoder(input_t1c__, is_training=is_training, drop_rate=drop_rate, name='ce_t1c_abs')
            feature_t1___ = feature_encoder(input_t1___, is_training=is_training, drop_rate=drop_rate, name='ce_t1_abs')
            feature_t2___ = feature_encoder(input_t2___, is_training=is_training, drop_rate=drop_rate, name='ce_t2_abs')

            feature_flair_s1 = feature_flair['s1']
            feature_flair_s2 = feature_flair['s2']
            feature_flair_s3 = feature_flair['s3']
            feature_flair_s4 = feature_flair['s4']

            feature_t1c___s1 = feature_t1c__['s1']
            feature_t1c___s2 = feature_t1c__['s2']
            feature_t1c___s3 = feature_t1c__['s3']
            feature_t1c___s4 = feature_t1c__['s4']

            feature_t1____s1 = feature_t1___['s1']
            feature_t1____s2 = feature_t1___['s2']
            feature_t1____s3 = feature_t1___['s3']
            feature_t1____s4 = feature_t1___['s4']

            feature_t2____s1 = feature_t2___['s1']
            feature_t2____s2 = feature_t2___['s2']
            feature_t2____s3 = feature_t2___['s3']
            feature_t2____s4 = feature_t2___['s4']

            feature_share_c1_concat = tf.concat(
                [feature_flair_s1, feature_t1c___s1, feature_t1____s1, feature_t2____s1], axis=-1, name='concat_c1_abs')
            feature_share_c1_attmap = layers.general_conv3d(feature_share_c1_concat, 4, pad_type='reflect',
                                                            name="att_c1_abs", act_type=None)
            feature_share_c1_attmap = tf.nn.sigmoid(feature_share_c1_attmap)
            feature_share_c1 = tf.concat([tf.multiply(feature_flair_s1, tf.tile(
                tf.expand_dims(feature_share_c1_attmap[:, :, :, :, 0], axis=-1), tf.constant([1, 1, 1, 1, 16]))),
                                          tf.multiply(feature_t1c___s1, tf.tile(
                                              tf.expand_dims(feature_share_c1_attmap[:, :, :, :, 1], axis=-1),
                                              tf.constant([1, 1, 1, 1, 16]))),
                                          tf.multiply(feature_t1____s1, tf.tile(
                                              tf.expand_dims(feature_share_c1_attmap[:, :, :, :, 2], axis=-1),
                                              tf.constant([1, 1, 1, 1, 16]))),
                                          tf.multiply(feature_t2____s1, tf.tile(
                                              tf.expand_dims(feature_share_c1_attmap[:, :, :, :, 3], axis=-1),
                                              tf.constant([1, 1, 1, 1, 16])))], axis=-1)

            feature_share_c2_concat = tf.concat(
                [feature_flair_s2, feature_t1c___s2, feature_t1____s2, feature_t2____s2], axis=-1, name='concat_c2_abs')
            feature_share_c2_attmap = layers.general_conv3d(feature_share_c2_concat, 4, pad_type='reflect',
                                                            name="att_c2_abs", act_type=None)
            feature_share_c2_attmap = tf.nn.sigmoid(feature_share_c2_attmap)
            feature_share_c2 = tf.concat([tf.multiply(feature_flair_s2, tf.tile(
                tf.expand_dims(feature_share_c2_attmap[:, :, :, :, 0], axis=-1), tf.constant([1, 1, 1, 1, 32]))),
                                          tf.multiply(feature_t1c___s2, tf.tile(
                                              tf.expand_dims(feature_share_c2_attmap[:, :, :, :, 1], axis=-1),
                                              tf.constant([1, 1, 1, 1, 32]))),
                                          tf.multiply(feature_t1____s2, tf.tile(
                                              tf.expand_dims(feature_share_c2_attmap[:, :, :, :, 2], axis=-1),
                                              tf.constant([1, 1, 1, 1, 32]))),
                                          tf.multiply(feature_t2____s2, tf.tile(
                                              tf.expand_dims(feature_share_c2_attmap[:, :, :, :, 3], axis=-1),
                                              tf.constant([1, 1, 1, 1, 32])))], axis=-1)

            feature_share_c3_concat = tf.concat(
                [feature_flair_s3, feature_t1c___s3, feature_t1____s3, feature_t2____s3], axis=-1, name='concat_c3_abs')
            feature_share_c3_attmap = layers.general_conv3d(feature_share_c3_concat, 4, pad_type='reflect',
                                                            name="att_c3_abs", act_type=None)
            feature_share_c3_attmap = tf.nn.sigmoid(feature_share_c3_attmap)
            feature_share_c3 = tf.concat([tf.multiply(feature_flair_s3, tf.tile(
                tf.expand_dims(feature_share_c3_attmap[:, :, :, :, 0], axis=-1), tf.constant([1, 1, 1, 1, 64]))),
                                          tf.multiply(feature_t1c___s3, tf.tile(
                                              tf.expand_dims(feature_share_c3_attmap[:, :, :, :, 1], axis=-1),
                                              tf.constant([1, 1, 1, 1, 64]))),
                                          tf.multiply(feature_t1____s3, tf.tile(
                                              tf.expand_dims(feature_share_c3_attmap[:, :, :, :, 2], axis=-1),
                                              tf.constant([1, 1, 1, 1, 64]))),
                                          tf.multiply(feature_t2____s3, tf.tile(
                                              tf.expand_dims(feature_share_c3_attmap[:, :, :, :, 3], axis=-1),
                                              tf.constant([1, 1, 1, 1, 64])))], axis=-1)

            feature_share_c4_concat = tf.concat(
                [feature_flair_s4, feature_t1c___s4, feature_t1____s4, feature_t2____s4], axis=-1, name='concat_c4_abs')
            feature_share_c4_attmap = layers.general_conv3d(feature_share_c4_concat, 4, pad_type='reflect',
                                                            name="att_c4_abs", act_type=None)
            feature_share_c4_attmap = tf.nn.sigmoid(feature_share_c4_attmap)
            feature_share_c4 = tf.concat([tf.multiply(feature_flair_s4, tf.tile(
                tf.expand_dims(feature_share_c4_attmap[:, :, :, :, 0], axis=-1), tf.constant([1, 1, 1, 1, 128]))),
                                          tf.multiply(feature_t1c___s4, tf.tile(
                                              tf.expand_dims(feature_share_c4_attmap[:, :, :, :, 1], axis=-1),
                                              tf.constant([1, 1, 1, 1, 128]))),
                                          tf.multiply(feature_t1____s4, tf.tile(
                                              tf.expand_dims(feature_share_c4_attmap[:, :, :, :, 2], axis=-1),
                                              tf.constant([1, 1, 1, 1, 128]))),
                                          tf.multiply(feature_t2____s4, tf.tile(
                                              tf.expand_dims(feature_share_c4_attmap[:, :, :, :, 3], axis=-1),
                                              tf.constant([1, 1, 1, 1, 128])))], axis=-1)

            feature_share_c1 = layers.general_conv3d(feature_share_c1, n_base_filters, k_size=1, pad_type='reflect',
                                                     name='fusion_c1_abs')
            feature_share_c2 = layers.general_conv3d(feature_share_c2, n_base_filters * 2, k_size=1, pad_type='reflect',
                                                     name='fusion_c2_abs')
            feature_share_c3 = layers.general_conv3d(feature_share_c3, n_base_filters * 4, k_size=1, pad_type='reflect',
                                                     name='fusion_c3_abs')
            feature_share_c4 = layers.general_conv3d(feature_share_c4, n_base_filters * 8, k_size=1, pad_type='reflect',
                                                     name='fusion_c4_abs')

            mask_de_input_prs = {'e1_out': feature_prs['s1'], 'e2_out': feature_prs['s2'], 'e3_out': feature_prs['s3'],
                                 'e4_out': feature_prs['s4'], }
            mask_de_input_abs = {
                'e1_out': feature_share_c1,
                'e2_out': feature_share_c2,
                'e3_out': feature_share_c3,
                'e4_out': feature_share_c4,
            }

            seg_pred_prs, seg_logit_prs, d1_out_prs = mask_decoder(mask_de_input_prs, num_cls, name='mask_de_prs')
            seg_pred_abs, seg_logit_abs, d1_out_abs = mask_decoder(mask_de_input_abs, num_cls, name='mask_de_abs')

            x_flat_prs, x_l1_prs, x_l2_prs, feat_prs = metric_layer(d1_out_prs, 'metric_prs')
            x_flat_abs, x_l1_abs, x_l2_abs, feat_abs = metric_layer(d1_out_abs, 'metric_abs')

            return {
                'feature_prs': feature_prs,
                'feature_flair': feature_flair,
                'feature_t1c__': feature_t1c__,
                'feature_t1___': feature_t1___,
                'feature_t2___': feature_t2___,
                'seg_pred_prs': seg_pred_prs,
                'seg_logit_prs': seg_logit_prs,
                'seg_pred_abs': seg_pred_abs,
                'seg_logit_abs': seg_logit_abs,
                'd1_out_prs': d1_out_prs,
                'd1_out_abs': d1_out_abs,
                'feat_prs': feat_prs,
                'feat_abs': feat_abs,
            }


def metric_layer(input, name):
    with tf.variable_scope(name):
        x_flat = tf.layers.flatten(tf.layers.average_pooling3d(inputs=input, pool_size=(40,40,40), strides=1))
        x_l1 = layers.linear(x_flat, feat_dim*2, norm_type=None, act_type='relu', name='linear_0')
        x_l2 = layers.linear(x_l1, feat_dim, norm_type=None, act_type=None, name='linear_1')
        x_norm = normalize_layer(x_l2)

        return x_flat, x_l1, x_l2, x_norm


def normalize_layer(x, power=2):
    norm = tf.pow(tf.reduce_sum(tf.pow(x, power), axis=-1, keep_dims=True), 1./power)
    out = tf.divide(x, norm)

    return out


def feature_encoder(input, is_training=True, drop_rate=0.3, name='feature_encoder'):
    with tf.variable_scope(name):
        e1_c1 = layers.general_conv3d(input, n_base_filters, pad_type='reflect', name="e1_c1")
        e1_c2 = layers.general_conv3d(e1_c1, n_base_filters, pad_type='reflect', name="e1_c2", drop_rate=drop_rate, is_training=is_training)
        e1_c3 = layers.general_conv3d(e1_c2, n_base_filters, pad_type='reflect', name="e1_c3")
        e1_out = e1_c1 + e1_c3

        e2_c1 = layers.general_conv3d(e1_out, n_base_filters * 2, s=2, pad_type='reflect', name="e2_c1")
        e2_c2 = layers.general_conv3d(e2_c1, n_base_filters * 2, pad_type='reflect', name="e2_c2", drop_rate=drop_rate, is_training=is_training)
        e2_c3 = layers.general_conv3d(e2_c2, n_base_filters * 2, pad_type='reflect', name="e2_c3")
        e2_out = e2_c1 + e2_c3

        e3_c1 = layers.general_conv3d(e2_out, n_base_filters * 4, s=2, pad_type='reflect', name="e3_c1")
        e3_c2 = layers.general_conv3d(e3_c1, n_base_filters * 4, pad_type='reflect', name="e3_c2", drop_rate=drop_rate, is_training=is_training)
        e3_c3 = layers.general_conv3d(e3_c2, n_base_filters * 4, pad_type='reflect', name="e3_c3")
        e3_out = e3_c1 + e3_c3

        e4_c1 = layers.general_conv3d(e3_out, n_base_filters * 8, s=2, pad_type='reflect', name="e4_c1")
        e4_c2 = layers.general_conv3d(e4_c1, n_base_filters * 8, pad_type='reflect', name="e4_c2", drop_rate=drop_rate, is_training=is_training)
        e4_c3 = layers.general_conv3d(e4_c2, n_base_filters * 8, pad_type='reflect', name="e4_c3")
        e4_out = e4_c1 + e4_c3

        return {
            's1':e1_out,
            's2':e2_out,
            's3':e3_out,
            's4':e4_out,
        }


def mask_decoder(input, num_cls, name='mask_decoder'):
    e4_out = input['e4_out']
    e3_out = input['e3_out']
    e2_out = input['e2_out']
    e1_out = input['e1_out']

    with tf.variable_scope(name):
        d3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(e4_out)
        d3_c1 = layers.general_conv3d(d3, n_base_filters * 4, pad_type='reflect', name="d3_c1")
        d3_cat = tf.concat([d3_c1, e3_out], axis=-1)
        d3_c2 = layers.general_conv3d(d3_cat, n_base_filters * 4, pad_type='reflect', name="d3_c2")
        d3_out = layers.general_conv3d(d3_c2, n_base_filters * 4, k_size=1, pad_type='reflect', name="d3_out")

        d2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(d3_out)
        d2_c1 = layers.general_conv3d(d2, n_base_filters * 2, pad_type='reflect', name="d2_c1")
        d2_cat = tf.concat([d2_c1, e2_out], axis=-1)
        d2_c2 = layers.general_conv3d(d2_cat, n_base_filters * 2, pad_type='reflect', name="d2_c2")
        d2_out = layers.general_conv3d(d2_c2, n_base_filters * 2, k_size=1, pad_type='reflect', name="d2_out")

        d1 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(d2_out)
        d1_c1 = layers.general_conv3d(d1, n_base_filters, pad_type='reflect', name="d1_c1")
        d1_cat = tf.concat([d1_c1, e1_out], axis=-1)
        d1_c2 = layers.general_conv3d(d1_cat, n_base_filters, pad_type='reflect', name="d1_c2")
        d1_out = layers.general_conv3d(d1_c2, n_base_filters, k_size=1, pad_type='reflect', name="d1_out")

        seg_logit = layers.general_conv3d(d1_out, num_cls, k_size=1, pad_type='reflect', name='seg_logit', norm_type=None, act_type=None)
        seg_pred = tf.nn.softmax(seg_logit, name='seg_pred')

        return seg_pred, seg_logit, d1_out


def mlp(style, name='MLP'):
    channel = mlp_ch
    with tf.variable_scope(name):
        x = layers.linear(style, channel, name='linear_0')
        x = tf.nn.relu(x)

        x = layers.linear(x, channel, name='linear_1')
        x = tf.nn.relu(x)

        mu = layers.linear(x, channel, name='mu')
        sigma = layers.linear(x, channel, name='sigma')

        mu = tf.reshape(mu, shape=[-1, 1, 1, 1, channel])
        sigma = tf.reshape(sigma, shape=[-1, 1, 1, 1, channel])

        return mu, sigma
