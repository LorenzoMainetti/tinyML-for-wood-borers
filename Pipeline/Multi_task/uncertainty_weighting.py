from keras.initializers.initializers import Constant
from keras.layers import Layer

from keras import backend as K
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy


class CustomMultiLossLayer(Layer):
    """
    Uncertainty Weights:
    A loss layer that calculates the weighted sum of losses for MTL.

    References
    ----------
    [1] Multi-Task Learning Using Uncertainty to Weight
    Losses for Scene Geometry and Semantics (CVPR 2018)
    """

    def __init__(self, nb_outputs=2, **kwargs):
        self.log_vars = None
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0

        precision1 = K.exp(-self.log_vars[0][0])
        loss1 = BinaryCrossentropy().call(ys_true[0], ys_pred[0])
        loss += K.sum(precision1 * loss1 + self.log_vars[0][0])

        precision2 = K.exp(-self.log_vars[1][0])
        loss2 = SparseCategoricalCrossentropy().call(ys_true[1], ys_pred[1])
        loss += K.sum(precision2 * loss2 + self.log_vars[1][0])

        return K.mean(loss)

    def call(self, inputs, *args, **kwargs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        return inputs[-2], inputs[-1]


