from typing import List, Dict, Tuple

import tensorflow as tf
from keras import layers
from keras.layers import Layer, Dense, Dropout

from custom_layers_utils import has_arg


class MultiGateMixtureOfExperts(Layer):
    """
    Multi-Gate Mixture of Experts - multitask learning model that allows
    automatically learn measure of relationship between tasks and adapts so that
    more relevant are able to share more information. Every task tower starts from
    its own Mixture of Expert layer that weights competency of each expert for a
    given task. Task specific layers are built on top of moe layers.

    Parameters
    ----------
    experts_layers: list of tensorflow.keras.Layer
      List of layers, where each layer represent corresponding Expert. Note that
      user can build composite layers, so that each layer would represent block of
      layers or the whole network ( this can be done through subclassing of
      tensorflow.keras.Layer).

    task_specific_layers: list of tensorflow.keras.Layer
      List of layers, where each layer represents part of the network that
      specializes in the specific task. Note that user can provide composite layer
      which can be a block of layers or whole neural network (this can be done
      through subclassing of tensorflow.keras.Layer)

    moe_dropout: bool, optional (Default=False)
      If True, then in the training stage experts are randomly dropped and expert
      competence probabilities are re-normalized in each of the MOE layers.

    moe_dropout_rate: float, optional (default=0.1)
      Probability that a single expert can be dropped in each of MOE layers.

    base_layer: tf.keras.layers.Layer, optional (default=None)
      User defined layer that preprocesses input (for instance splits numeric
      and categorical features and then normalizes numeric ones and creates
      embeddings for categorical ones)

    base_expert_prob_layer: tf.keras.layers.Layer, optional (Default=None)
        Layer that extracts features from inputs.

    References
    ----------
    [1] Modeling Task Relationships in Multi_task Learning with
        Multi-gate Mixture-of-Experts, Jiaqi Ma1 et al, 2018 KDD
    """

    def __init__(self,
                 expert_layers: List[Layer],
                 task_layers: List[Layer],
                 moe_layers: List[Layer] = None,
                 moe_dropout: bool = False,
                 moe_dropout_rate: float = 0.1,
                 base_layer: Layer = None,
                 base_expert_prob_layer: Layer = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_layer = base_layer
        self.task_layers = task_layers
        self.n_tasks = len(task_layers)
        self.expert_layers = expert_layers
        self.moe_dropout = moe_dropout
        self.moe_dropout_rate = moe_dropout_rate
        self.base_expert_prob_layer = base_expert_prob_layer
        self.moe_layers = moe_layers if moe_layers else self._build_moe_layers()

    def _build_moe_layers(self) -> List[Layer]:
        """Builds Mixture of Experts Layers if they are not provided by the user"""
        moe_layers = []
        for i in range(self.n_tasks):
            moe_layers.append(MixtureOfExpertsLayer(self.expert_layers,
                                                    self.moe_dropout,
                                                    self.moe_dropout_rate,
                                                    self.base_expert_prob_layer))
        return moe_layers

    def call(self, inputs, training):
        """
        Forward pass of the Multi-Gate Mixture of Experts model.

        Parameters
        ----------
        inputs: np.array or tf.Tensor
          Input to the model

        training: bool
          If True runs model in training mode, otherwise in prediction
          mode.

        Returns
        -------
        outputs: list of tf.Tensor
          Outputs of forward pass for each task
        """
        outputs = []
        if self.base_layer:
            if has_arg(self.base_layer, "training"):
                inputs = self.base_layer(inputs, training)
            else:
                inputs = self.base_layer(inputs)
        moes = [moe(inputs, training) for moe in self.moe_layers]
        for task, moe in zip(self.task_layers, moes):
            if has_arg(task, "training"):
                outputs.append(task(moe, training))
            else:
                outputs.append(task(moe))
        return outputs

    def get_config(self) -> Dict:
        """ Get configuration of the Multi-Gate Mixture of Experts """
        base_config = super().get_config()
        return {**base_config,
                "base_layer": layers.serialize(self.base_layer) if self.base_layer else None,
                "task_layers": [layers.serialize(l) for l in self.task_layers],
                "moe_layers": [layers.serialize(l) for l in self.moe_layers],
                "moe_dropout": self.moe_dropout,
                "moe_dropout_rate": self.moe_dropout_rate
                }


class ExpertUtilizationDropout(Layer):
    """
    Helper layer for Mixture of Experts Layer, allows to drop some of the experts
    and then renormalize other experts, so that we still have probability density
    function for expert utilization.

    Parameters
    ----------
    dropout_rate: float (between 0 and 1), optional (Default=0.1)
      Probability of drop

    References
    ----------
    [1] Recommending What Video to Watch Next: A Multitask Ranking System, 2019.
        Zhe Zhao
    """

    def __init__(self,
                 dropout_rate: float = 0.1,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.dropout_rate = dropout_rate

    def build(self,
              batch_input_shape: Tuple[int]
              ) -> None:
        """
        Builds main component of the layer by creating dropout layer that has
        the same binary mask for each element in the batch.

        Parameters
        ----------
        batch_input_shape: tuple
          Shape of the inputs
        """
        self.drop_layer = Dropout(self.dropout_rate,
                                  noise_shape=(1, *batch_input_shape[1:]),
                                  **self.kwargs
                                  )
        super().build(batch_input_shape)

    def call(self, inputs, training):
        """
        Defines forward pass for the layer, by dropping experts and then
        renormalizing utilization probabilities for remaining experts.

        Parameters
        ----------
        inputs: np.array or tf.Tensor
          Inputs to the layer (can be numpy array, tf.Tensor)

        training: bool
          True if layer is called in training mode, False otherwise

        Returns
        -------
        : tf.Tensor
          Renormalized probabilities assigned to each expert
        """
        expert_probs_drop = self.drop_layer(inputs, training)
        # add epsilon to prevent division by zero when all experts are dropped
        normalizer = tf.add(tf.reduce_sum(expert_probs_drop, keepdims=True, axis=1),
                            tf.keras.backend.epsilon()
                            )
        return expert_probs_drop / normalizer

    def get_config(self) -> Dict:
        """ Config of the ExpertUtilizationDropout layer """
        base_config = super().get_config()
        return {**base_config, "dropout_rate": self.dropout_rate}


class OneGateMixtureOfExperts(MultiGateMixtureOfExperts):
    """
    One-Gate Mixture of Experts - similar to Multi-Gate Mixture of Experts.
    The only difference is that every task shares the same MOE layer, while in
    Multi-Gate there is specific MOE layer for each task.

    Parameters
    ----------
    experts_layers: list of tensorflow.keras.Layer
      List of layers, where each layer represent corresponding Expert. Note that
      user can build composite layers, so that each layer would represent block of
      layers or the whole network ( this can be done through subclassing of
      tensorflow.keras.Layer).

    task_specific_layers: list of tensorflow.keras.Layer
      List of layers, where each layer represents part of the network that
      specializes in the specific task. Note that user can provide composite layer
      which can be a block of layers or whole neural network (this can be done
      through subclassing of tensorflow.keras.Layer)

    moe_dropout: bool, optional (Default=False)
      If True, then in the training stage experts are randomly dropped and expert
      competence probabilities are re-normalized in the MOE layer.

    moe_dropout_rate: float, optional (default=0.1)
      Probability that a single expert can be dropped in the MOE layer.

    base_layer: tensorflow.keras.Layer, optional (default=None)
      User defined layer that preprocesses input (for instance splits numeric
      and categorical features and then normalizes numeric ones and creates
      embeddings for categorical ones)

    base_expert_prob_layer: tf.keras.layers.Layer, optional (Default=None)
        Layer that extracts features from inputs.

    References
    ----------
    [1] Modeling Task Relationships in Multi_task Learning with
        Multi-gate Mixture-of-Experts, Jiaqi Ma1 et al, 2018 KDD
    """

    def _build_moe_layers(self):
        """Builds Mixture of Experts Layers if they are not provided by the user"""
        return [MixtureOfExpertsLayer(self.expert_layers, self.moe_dropout, self.moe_dropout_rate)]

    def call(self, inputs, training):
        """
        Forward pass of the One-Gate Mixture of Experts Model

        Parameters
        ----------
        inputs: np.array or tf.Tensor
          Input to the model

        training: bool
          True during training, False otherwise

        Returns
        -------
        outputs: list of tf.Tensor
          Outputs of forward pass for each task
        """
        outputs = []
        if self.base_layer:
            if has_arg(self.base_layer, "training"):
                inputs = self.base_layer(inputs, training)
            else:
                inputs = self.base_layer(inputs)
        moe = [moe(inputs, training) for moe in self.moe_layers][0]
        for task in self.task_layers:
            if has_arg(task, "training"):
                outputs.append(task(moe, training))
            else:
                outputs.append(task(moe))
        return outputs


class MixtureOfExpertsLayer(Layer):
    """
    Mixture of Experts Layer

    Parameters
    ----------
    expert_layers: List of Layers
        List of experts, each expert is expected to be a instance of Layer class.

    add_dropout: bool, optional (Default=False)
        Adds dropout after softmax layer, helps to avoid collapse of the experts
        and their imbalanced utilization. Was used by Zhe Zhao in [1].

    dropout_rate: float (between 0 and 1), optional (Default=0.1)
        Fraction of the experts to drop.

    base_expert_prob_layer: tf.keras.Layer, optional (Default=None)
        Layer that extracts features from inputs.

    References
    ----------
    [1] Recommending What Video to Watch Next: A Multitask Ranking System, 2019.
        Zhe Zhao
    [2] Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
        Layer, 2017. Noam Shazeer, Azalia Mirhoseini.
    """

    def __init__(self,
                 expert_layers: List[Layer],
                 add_dropout: bool = False,
                 dropout_rate: float = 0.1,
                 base_expert_prob_layer: Layer = None,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.expert_layers = expert_layers
        self.n_experts = len(expert_layers)
        self.add_dropout = add_dropout
        self.dropout_rate = dropout_rate
        self.base_expert_prob_layer = base_expert_prob_layer
        self.expert_probs = Dense(self.n_experts, "softmax", **self.kwargs)
        if add_dropout:
            self.drop_expert_layer = ExpertUtilizationDropout(self.dropout_rate)

    def call(self, inputs, training):
        """
        Defines set of computations performed in the MOE layer.
        MOE layer can accept single tensor (in this case it assumes the same
        input for every expert) or collection/sequence of tensors
        (in this case it assumes every tensor corresponds to its own expert)

        Parameters
        ----------
        inputs: np.array, tf.Tensor, list/tuple of np.arrays or  tf.Tensors
          Inputs to the MOE layer

        training: bool
          True if layer is called in training mode, False otherwise

        Returns
        -------
        moe_output: tf.Tensor
          Output of mixture of experts layers ( linearly weighted output of expert
          layers).
        """
        # compute each expert output (optionally pass training argument,
        # since some experts may contain training arg, some may not.
        experts_output = []
        for expert in self.expert_layers:
            if has_arg(expert, "training"):
                experts_output.append(expert(inputs, training))
            else:
                experts_output.append(expert(inputs))

        # compute probability of expert (degree of expert utilization) for given
        # input set
        if self.base_expert_prob_layer:
            inputs = self.base_expert_prob_layer(inputs)
        expert_utilization_prob = self.expert_probs(inputs)
        if self.add_dropout:
            expert_utilization_prob = self.drop_expert_layer(expert_utilization_prob,
                                                             training)

        # compute weighted output of experts
        moe_output = 0
        for i, expert_output in enumerate(experts_output):
            moe_output += (expert_output
                           * tf.expand_dims(expert_utilization_prob[:, i], axis=-1)
                           )
        return moe_output

    def get_config(self) -> Dict:
        """ Config of the MOE layer """
        base_config = super().get_config()
        return {**base_config,
                "add_dropout": self.add_dropout,
                "dropout_rate": self.dropout_rate,
                "expert_layers": [tf.keras.layers.serialize(layer) for
                                  layer in self.expert_layers],
                "base_expert_prob_layer": tf.keras.layers.serialize(
                    self.base_expert_prob_layer) if self.base_expert_prob_layer else None
                }
