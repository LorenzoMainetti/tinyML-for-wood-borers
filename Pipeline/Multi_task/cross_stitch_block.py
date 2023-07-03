from typing import List

import tensorflow as tf
from keras.initializers.initializers import RandomUniform
from keras.layers import Layer


class CrossStitch(Layer):
    """
    Cross-stitch block

    References
    ----------
    [1] Cross-stitch Networks for Multi_task Learning, (2017)
    Ishan Misra et al
    """

    def build(self,
              batch_input_shape: List[tf.TensorShape]
              ) -> None:
        self.num_tasks = len(batch_input_shape)

        # initialize using random uniform distribution as suggested in Section 5.1
        self.cross_stitch_kernel = self.add_weight(shape=(self.num_tasks, self.num_tasks),
                                                   initializer=RandomUniform(0., 1.),
                                                   trainable=True,
                                                   name="cross_stitch_kernel")

        # normalize, so that each row will be convex linear combination,
        # here we follow recommendation in paper ( see section 5.1 )
        normalizer = tf.reduce_sum(self.cross_stitch_kernel,
                                   keepdims=True,
                                   axis=0)
        self.cross_stitch_kernel.assign(self.cross_stitch_kernel / normalizer)

    def call(self, inputs, **kwargs):
        """
        called by TensorFlow when the model gets build.
        Returns a stacked tensor with num_tasks channels in the 0 dimension,
        which need to be unstacked.
        """
        if len(inputs) != self.num_tasks:
            # should not happen
            raise ValueError()

        out_values = []
        for this_task in range(self.num_tasks):
            this_weight = self.cross_stitch_kernel[this_task, this_task]
            out = tf.math.scalar_mul(this_weight, inputs[this_task])
            for other_task in range(self.num_tasks):
                if this_task == other_task:
                    continue  # already weighted!
                other_weight = self.cross_stitch_kernel[this_task, other_task]
                out += tf.math.scalar_mul(other_weight, inputs[other_task])
            out_values.append(out)

        return out_values

    def compute_output_shape(self, input_shape):
        return [self.num_tasks] + input_shape

    def get_config(self):
        """implemented so keras can save the model to json/yml"""
        config = {
            "num_tasks": self.num_tasks
        }
        base_config = super(CrossStitch, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))

