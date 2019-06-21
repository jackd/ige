from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class InputSink(tf.keras.layers.Layer):
    """
    This is so dumb.
    
    Necessary because keras has issues during evaluation if inputs aren't used.
    """
    def __init__(self, num_used=1):
        self.num_used = num_used
        super(InputSink, self).__init__()
    
    def get_config(self):
        return dict(num_used=self.num_used)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:self.num_used]
    
    def call(self, inputs):
        return [tf.identity(i) for i in inputs[:self.num_used]]