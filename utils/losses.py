"""
SPDX-License-Identifier: MIT
Copyright © 2017 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/ssd_detectors
"""


import numpy as np
import tensorflow as tf
import keras.backend as K


def square_loss(y_true, y_pred):
    loss = tf.square(y_true - y_pred)
    return tf.reduce_sum(loss, axis=-1)

def absolute_loss(y_true, y_pred):
    loss = tf.abs(y_true - y_pred)
    return tf.reduce_sum(loss, axis=-1)

def smooth_l1_loss(y_true, y_pred):
    """Compute L1-smooth loss.

    # Arguments
        y_true: Ground truth, tensor of shape (..., n)
        y_pred: Prediction, tensor of shape (..., n)

    # Returns
        loss: Smooth L1-smooth loss, tensor of shape (...)

    # References
        [Fast R-CNN](https://arxiv.org/abs/1504.08083)
    """
    abs_loss = tf.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred)**2
    loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return tf.reduce_sum(loss, axis=-1)

def shrinkage_loss(y_true, y_pred, a=10.0, c=0.2):
    """Compute Shrikage Loss.

    # Arguments
        y_true: Ground truth, tensor of shape (..., n)
        y_pred: Prediction, tensor of shape (..., n)

    # Returns
        loss: Smooth L1-smooth loss, tensor of shape (...)

    # References
        [Deep Regression Tracking with Shrinkage Loss](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf)
    """
    l = tf.abs(y_true - y_pred)
    loss = tf.square(l) / (1 + tf.exp(a*(c-l)))
    return tf.reduce_sum(loss, axis=-1)

def dynamic_shrinkage_loss(y_true, y_pred, mask=None, a=5.0, c=0.5, autoscale=True, mean_all=False, reduce=True,
                          with_certainty=False):
    """Dynamically scaled version of the Shrinkage Loss.

    # Arguments
        y_true, y_pred: float tensor of shape (n,c)
        mask: float tensor with binary values of shape (n)
        a: float parameter from Shrinkage Loss
        c: float parameter form Shrinkage Loss
        autoscale: boolean
        mean_all: boolean
        reduce: boolean
    # Retrun
        mean_abs:
        loss:
        [certainty]:
    """

    eps = 1e-6

    if mask is not None:
        mask = mask[...,None]
    else:
        mask = K.ones_like(y_true[...,0:1])
    num_pos = K.sum(mask)

    l = tf.abs(y_true - y_pred)

    abs_err = l

    if mean_all:
        m = K.mean(l, axis=-2, keepdims=True)
    else:
        m = K.sum(l*mask, axis=-2, keepdims=True) / (num_pos + eps)

    m = tf.stop_gradient(tf.clip_by_value(m, eps, 1e5))

    if autoscale:
        l = 0.5 * l / (m + eps)

    f = 1 / (1 + tf.exp(a*(c-l)))

    loss = f * l
    #loss = f * K.square(l)
    #loss = - f * K.log(l+eps)

    loss = tf.reduce_sum(loss, axis=-1, keepdims=True) * mask
    mean_abs = tf.reduce_mean(abs_err, axis=-1, keepdims=True) * mask

    if reduce:
        loss = K.sum(loss) / (num_pos + eps)
        mean_abs = K.sum(mean_abs) / (num_pos + eps)

    if with_certainty:
        certainty = K.exp(abs_err/(m+eps) * np.log(0.5))
        certainty = K.min(certainty, axis=-1, keepdims=True)
        certainty = K.stop_gradient(certainty)
        # notes:
        #     np.log(0.5) = -0.6931471805599453
        #     certainty > 0.5 if error is less then average error
        #     certainty < 0.5 if error is greater then average error
        return mean_abs, loss, certainty

    return mean_abs, loss

def softmax_loss(y_true, y_pred):
    """Compute cross entropy loss aka softmax loss.

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).

    # Returns
        loss: Softmax loss, tensor of shape (...)

    """
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.-eps)
    loss = - y_true * K.log(y_pred)
    return tf.reduce_sum(loss, axis=-1)

def cross_entropy_loss(y_true, y_pred):
    """Compute binary cross entropy loss.
    """
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.-eps)
    #loss = - y_true*K.log(y_pred) - (1.-y_true)*K.log(1.-y_pred)
    pt = tf.where(tf.equal(y_true, 1.), y_pred, 1.-y_pred)
    loss = - K.log(pt)
    return tf.reduce_sum(loss, axis=-1)

def focal_loss(y_true, y_pred, gamma=2., alpha=1.):
    """Compute binary focal loss.
    
    # Arguments
        y_true: Ground truth, tensor of shape (..., n)
        y_pred: Prediction, tensor of shape (..., n).
    
    # Returns
        loss: Focal loss, tensor of shape (...)

    # References
        [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
    """
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.-eps)
    #loss = - K.pow(1-y_pred, gamma) * y_true*K.log(y_pred) - K.pow(y_pred, gamma) * (1-y_true)*K.log(1-y_pred)
    pt = tf.where(tf.equal(y_true, 1.), y_pred, 1.-y_pred)
    loss = - K.pow(1.-pt, gamma) * K.log(pt)
    loss = alpha * loss
    return tf.reduce_sum(loss, axis=-1)

def reduced_focal_loss(y_true, y_pred, gamma=2., alpha=1., th=0.5):
    """Compute binary reduced focal loss.
    
    # Arguments
        y_true: Ground truth, tensor of shape (..., n)
        y_pred: Prediction, tensor of shape (..., n)

    # Returns
        loss: Reduced focal loss, tensor of shape (...)

    # References
        [Reduced Focal Loss: 1st Place Solution to xView object detection in Satellite Imagery](https://arxiv.org/abs/1903.01347)
    """
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.-eps)
    pt = tf.where(tf.equal(y_true, 1.), y_pred, 1.-y_pred)
    fr = tf.where(tf.less(pt, 1.-th), 1., K.pow((1.-pt)/th, gamma))
    loss = - fr * K.log(pt)
    loss = alpha * loss
    return tf.reduce_sum(loss, axis=-1)


def ciou_loss(y_true, y_pred, variant='diou'):
    '''Conpute Distance-IoU loss.

    # Arguments
        y_true: Ground truth bounding boxes, tensor of shape (..., 4)
        y_pred: Predicted bounding boxes, tensor of shape (..., 4)
        variant: 'diou', 'ciou', 'logciou'

    # Returns
        loss: Distance-IoU loss, tensor of shape (...)

    # Notes
        takes in a list of bounding boxes
        but can work for a single bounding box too
        bounding boxes are specified with (x_min, y_min, x_max, y_max)
        all the boundary cases such as bounding boxes of size 0 are handled
    
    # References
        [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
    
    # Source
        https://github.com/notabee/Distance-IoU-Loss-Faster-and-Better-Learning-for-Bounding-Box-Regression/blob/master/ciou.py
    '''
    mask = tf.cast(y_true != 0, dtype='float32')
    y_true = y_true * mask
    y_pred = y_pred * mask

    x1g, y1g, x2g, y2g = tf.unstack(y_true, axis=-1)
    x1, y1, x2, y2 = tf.unstack(y_pred, axis=-1)
    
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)
    
    # iou
    xA = tf.maximum(x1g, x1)
    yA = tf.maximum(y1g, y1)
    xB = tf.minimum(x2g, x2)
    yB = tf.minimum(y2g, y2)

    interArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(0.0, yB - yA + 1)

    boxAArea = (x2g - x1g +1) * (y2g - y1g +1)
    boxBArea = (x2 - x1 +1) * (y2 - y1 +1)

    iouk = interArea / (boxAArea + boxBArea - interArea + 1e-10)
    
    # distance
    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2)
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / (c + 1e-7)

    # aspect-ratio
    arctan = tf.atan(w_gt/(h_gt + 1e-10))-tf.atan(w_pred/(h_pred + 1e-10))
    v = (4 / (np.pi ** 2)) * tf.pow((tf.atan(w_gt/(h_gt + 1e-10))-tf.atan(w_pred/(h_pred + 1e-10))),2)
    S = 1 - iouk
    alpha = v / (S + v + 1e-10)
    w_temp = 2 * w_pred
    ar = (8 / (np.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)
    
    # calculate diou, ciou, ...
    if variant == 'diou':
        return 1-iouk + u
    elif variant == 'ciou':
        return 1-iouk + u + alpha*ar
    elif variant == 'logciou':
        # "I found that -log(IoU) is more stable and converge faster than (1-IoU)"
        return -tf.math.log(iouk) + u + alpha*ar
    else:
        return None

