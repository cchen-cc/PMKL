import tensorflow as tf


def _softmax_weighted_loss(pred, label, num_cls):
    '''
    calculate weighted cross-entropy loss, the weight is dynamic dependent on the data
    '''

    for i in xrange(num_cls):
        labeli = label[:,:,:,:,i]
        predi = pred[:,:,:,:,i]
        weighted = 1.0-(tf.reduce_sum(labeli) / tf.reduce_sum(label))
        if i == 0:
            raw_loss = -1.0 * weighted * labeli * tf.log(tf.clip_by_value(predi, 0.005, 1))
        else:
            raw_loss += -1.0 * weighted * labeli * tf.log(tf.clip_by_value(predi, 0.005, 1))

    loss = tf.reduce_mean(raw_loss)

    return loss

def softmax_loss(pred, label, num_cls):
    for i in xrange(num_cls):
        labeli = label[:,:,:,:,i]
        predi = pred[:,:,:,:,i]
        
        if i == 0:
            raw_loss = -1.0 * labeli * tf.log(tf.clip_by_value(predi, 0.005, 1))
        else:
            raw_loss += -1.0 * labeli * tf.log(tf.clip_by_value(predi, 0.005, 1))

    loss = raw_loss

    return loss


def _dice_loss_fun(pred, label, num_cls):
    '''
    calculate dice loss, - 2*interesction/union, with relaxed for gradients backpropagation
    '''
    dice = 0.0
    eps = 1e-7

    for i in xrange(num_cls):
        inse = tf.reduce_sum(pred[:, :, :, :, i] * label[:, :, :, :, i])
        l = tf.reduce_sum(pred[:, :, :, :, i] * pred[:, :, :, :, i])
        r = tf.reduce_sum(label[:, :, :, :, i])
        dice += 2.0 * inse/(l+r+eps)

    return 1.0 - 1.0 * dice / num_cls


def task_loss(pred, label, num_cls):

    ce_loss = _softmax_weighted_loss(pred, label, num_cls)
    dice_loss = _dice_loss_fun(pred, label, num_cls)

    return ce_loss, dice_loss


def kd_loss(student, teacher, hard_label, num_cls, temperature=10.0):
    soft_teacher = tf.nn.softmax(teacher / temperature)
    soft_student = tf.nn.softmax(student / temperature)

    kd_xentropy = softmax_loss(pred=soft_student, label=soft_teacher, num_cls=num_cls)

    return kd_xentropy