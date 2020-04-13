
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class TBPPDecodeAndCrop(Layer):
    # NOTES
    # requires tensorflow-addons
    # does not work with changing image size
    # limited to one class
    # provieds grayscale images to the recogition stage
    
    def __init__(self, prior_util, confidence_threshold=0.01, iou_threshold=0.45, top_k=200, output_size=(32, 256), **kwargs):
        self.prior_util = prior_util
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.output_size = output_size
        self.output_height, self.output_width = output_size
        
        super(TBPPDecodeAndCrop, self).__init__(**kwargs)
        
    def build(self, input_shape):
        image_size = self.prior_util.image_size
        
        priors_xy = self.prior_util.priors_xy / image_size
        priors_wh = self.prior_util.priors_wh / image_size
        priors_variances = self.prior_util.priors_variances
        
        self.priors_xy = tf.constant(priors_xy, dtype=K.floatx())
        self.priors_wh = tf.constant(priors_wh, dtype=K.floatx())
        self.priors_variances = tf.constant(priors_variances, dtype=K.floatx())
        
        priors_minmax = np.hstack([priors_xy-priors_wh/2, priors_xy+priors_wh/2])
        self.ref = tf.constant(priors_minmax[:,(0,1,2,1,2,3,0,3)], dtype=K.floatx()) # corner points
        
        super(TBPPDecodeAndCrop, self).build(input_shape)
        
    def call(self, x):
        # calculation is done with normalized sizes
        # x[0] image
        # x[1] local predictions
        #     4 mbox_loc + 8 mbox_quad + 5 mbox_rbox + 2 mbox_conf
        
        def for_each_sample(x):
            ## decoding and NMS
            y_pred = x[1]
            
            mask = y_pred[:,18] > self.confidence_threshold
            boxes_to_process = tf.boolean_mask(y_pred, mask)
            
            #priors = tf.boolean_mask(self.priors, mask)
            priors_xy = tf.boolean_mask(self.priors_xy, mask)
            priors_wh = tf.boolean_mask(self.priors_wh, mask)
            variances = tf.boolean_mask(self.priors_variances, mask)
            variances_xy = variances[:,0:2]
            variances_wh = variances[:,2:4]
            
            offsets = boxes_to_process[:,:4]
            boxes_xy = priors_xy + offsets[:,0:2] * variances_xy * priors_wh
            boxes_wh = priors_wh * tf.exp(offsets[:,2:4] * variances_wh)
            boxes_xy_min = boxes_xy - boxes_wh / 2.
            boxes_xy_max = boxes_xy + boxes_wh / 2.
            boxes_minmax = tf.concat((boxes_xy_min, boxes_xy_max), axis=-1)
            #boxes_minmax = tf.clip_by_value(boxes_minmax, 0.0, 1.0)
            
            scores = boxes_to_process[:,18]
            
            idxs = tf.image.non_max_suppression(boxes_minmax, scores,
                                                max_output_size=self.top_k, 
                                                iou_threshold=self.iou_threshold)
            
            good_boxes = tf.gather(boxes_to_process, idxs, axis=0)
            
            good_minmax = tf.gather(boxes_minmax, idxs, axis=0)
            
            ref = tf.boolean_mask(self.ref, mask)
            ref = tf.gather(ref, idxs, axis=0)
            priors_xy = tf.gather(priors_xy, idxs, axis=0)
            priors_wh = tf.gather(priors_wh, idxs, axis=0)
            variances_xy = tf.gather(variances_xy, idxs, axis=0)
            variances_wh = tf.gather(variances_wh, idxs, axis=0)
            
            offsets_quads = good_boxes[:,4:12]
            good_quads = ref + offsets_quads * tf.tile(priors_wh * variances_xy, (1,4))
            
            offsets_rboxs = good_boxes[:,12:17]
            good_rboxs = tf.concat((
                priors_xy + offsets_rboxs[:,0:2] * priors_wh * variances_xy,
                priors_xy + offsets_rboxs[:,2:4] * priors_wh * variances_xy,
                tf.exp(offsets_rboxs[:,4:5] * variances_wh[:,1:2]) * priors_wh[:,1:2]
            ),-1)
            
            good_confs = good_boxes[:,18:19]
            
            # we only have one class :)
            good_labels = tf.ones_like(good_confs)
            
            # 4 boxes + 8 quad + 5 rboxes + 1 confs + 1 labels
            good_boxes = tf.concat((
                good_minmax, 
                good_quads, 
                good_rboxs, 
                good_confs, 
                good_labels, 
            ), -1)
            
            ## cropping
            
            img = x[0]
            img = tf.image.rgb_to_grayscale(img)
            img = tf.expand_dims(img, 0)
            
            img_size = tf.cast(tf.shape(img)[1:3], 'float32')
            polys = good_boxes[:,4:12] * tf.tile(img_size, (4,))
            n = num_polys = tf.shape(polys)[0]
            
            h, w = self.output_height, self.output_width
            
            p = h * 0.05
            d = h/2
            pad_value = 0
            
            def crop():
                tl, tr, br, bl = polys[:,0:2], polys[:,2:4], polys[:,4:6], polys[:,6:8]
                
                box_h = (tf.norm(tl-bl, axis=-1) + tf.norm(tr-br, axis=-1)) / 2
                box_w = (tf.norm(tl-tr, axis=-1) + tf.norm(bl-br, axis=-1)) / 2
                
                w_mod = tf.clip_by_value(h*box_w/box_h, 0, w)
                
                x = tf.stack((tf.zeros_like(w_mod)+p, w_mod-p, w_mod-p, tf.zeros_like(w_mod)+p), axis=-1)
                y = tf.repeat(np.array([[p,p,h-p,h-p]], dtype='float32'), n, axis=0)
                u = polys[:,0::2]
                v = polys[:,1::2]
                
                ones, zeros = tf.ones((n, 4)), tf.zeros((n, 4))
                
                A = tf.concat([
                    tf.stack([x, y, ones, zeros, zeros, zeros, -x*u, -y*u], axis=-1),
                    tf.stack([zeros, zeros, zeros, x, y, ones, -x*v, -y*v], axis=-1),
                ], axis=1)
                
                b = tf.expand_dims(tf.concat((u,v), axis=-1), axis=-1)
                
                M = tf.squeeze(tf.linalg.solve(A, b), axis=-1)
                
                w_max = tf.reduce_max(w_mod)
                w_mod = tf.cast(w_mod, 'int32')
                
                def for_each_box(i, offset, words, offsets):
                    c = w_mod[i]
                    
                    word = tfa.image.transform(img, M[i], interpolation='BILINEAR', output_shape=(h,w_max))
                    word = tf.transpose(word, (0,2,1,3))
                    word = tf.concat([ word[0,:c], tf.ones((d,h,1))*pad_value ], axis=0)
                    words = words.write(i, word)
                    
                    offset = offset + tf.cast(c, 'float32') + d
                    offsets = offsets.write(i, offset)
                    
                    return i+1, offset, words, offsets
                
                words = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=n)
                offsets = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=n) 
                
                _, _, words, offsets = tf.while_loop(
                    lambda i, *_: tf.less(i, n), for_each_box, 
                    [0, -d/2, words, offsets])
                
                words = words.concat()
                offsets = offsets.stack()
                
                return words, offsets
            
            words, offsets = tf.cond( n > 0, crop,
                lambda: (tf.zeros((0,32,1)), tf.zeros(0)) )
            
            words = words[:w,:,0]
            words = tf.pad(words, ((0,w-tf.shape(words)[0]), (0,0)), constant_values=pad_value)
            words = tf.expand_dims(words, axis=-1)
            
            offsets = tf.expand_dims(offsets, axis=-1)
            
            good_boxes = tf.concat([good_boxes, offsets], axis=-1)
            good_boxes = tf.pad(good_boxes, ((0,self.top_k-tf.shape(good_boxes)[0]),(0,0)), constant_values=0)
            
            return words, good_boxes
        
        cropped_images, boxes = tf.map_fn(for_each_sample, x, dtype=('float32', 'float32'))
        
        return [cropped_images, boxes]

    def get_config(self):
        base_config = super(TBPPDecodeAndCrop, self).get_config()
        
        base_config['prior_util'] = self.prior_util
        base_config['confidence_threshold'] = self.confidence_threshold
        base_config['iou_threshold'] = self.iou_threshold
        base_config['top_k'] = self.top_k
        base_config['output_size'] = self.output_size
        
        return base_config

