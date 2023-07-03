from typing import List, Dict, Sequence, Tuple

import tensorflow as tf
from keras.layers import Layer

from custom_layers_utils import regularize_norm_diff, has_arg


class ConstrainedMTL(Layer):
    """
    Soft Parameter Sharing Multi-Task Learning.

    Parameters
    ----------
    mtl_layers: List of Layer instances
        List of layers, each layer corresponds to specific task

    l1_regularizer: float
        Strength of L1 penalty for difference in parameters between
        different layers

    l2_regularizer: float
        Strength of L2 penalty for difference in parameters between
        different layers

    References
    ----------
    [1] Low Resource Dependency Parsing: Cross-lingual Parameter Sharing in
    a Neural Network Parser. Duong, L., Cohn, T., Bird, S., & Cook, P. (2015)
    """

    def __init__(self,
                 mtl_layers: List[Layer],
                 l1_regularizer: float,
                 l2_regularizer: float,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.layers = mtl_layers
        self.l1_regularizer = l1_regularizer
        self.l2_regularizer = l2_regularizer

    def call(self, inputs, training):
        """
        Forward pass through constraining layer. Constraining layer can
        accept single tensor (in this case it assumes the same
        input for every expert) or collection/sequence of tensors
        (in this case it assumes every tensor corresponds to its own expert)

        Parameters
        ----------
        inputs: tf.Tensor, np.array or List/Tuple of tf.Tensors/np.arrays
            Input tensor

        training: bool
            True in case of training, False otherwise
        """
        # compute output of the layer
        outputs = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            if has_arg(layer, "training"):
                if isinstance(inputs, Sequence):
                    outputs[i] = layer(inputs[i], training)
                else:
                    outputs[i] = layer(inputs, training)
            else:
                outputs[i] = layer(inputs[i]) if isinstance(inputs, (List, Tuple)) else layer(inputs)

        # get all trainable variables from every column of MTL
        trainable_vars = [layer.trainable_variables for layer in self.layers]

        # add constraining loss
        sharing_loss = 0.
        if self.l2_regularizer > 0.:
            sharing_loss += self.l2_regularizer * regularize_norm_diff(trainable_vars, "L2")
        if self.l1_regularizer > 0.:
            sharing_loss += self.l1_regularizer * regularize_norm_diff(trainable_vars, "L1")

        # add sharing loss and return outputs
        self.add_loss(sharing_loss)
        return outputs

    def get_config(self) -> Dict:
        base_config = super().get_config()
        return {**base_config,
                "l1_regularizer": self.l1_regularizer,
                "l2_regularizer": self.l2_regularizer,
                "mtl_layers": [tf.keras.layers.serialize(l) for l in self.layers]
                }


