from PostTransformation import PostTransformConfig
import transformers.models.gpt2.modeling_tf_gpt2
import tensorflow as tf


class PositionalInvariantAddLayer(tf.keras.layers.Layer):

    def __init__(self, postShape, **kwargs):
        super().__init__(**kwargs)
        self.postShape = postShape
        self.postPastWeight = self.add_weight('Post-Past', self.postShape, tf.float32, tf.keras.initializers.Zeros(),
                                              trainable=True)

    def call(self, inputs, **kwargs):
        tf.print(tf.reduce_sum(self.postPastWeight))
        return inputs + self.postPastWeight


class PositionalInvariantTransformation(transformers.TFPreTrainedModel):
    config_class = PostTransformConfig.PostTransformConfig

    @property
    def dummy_inputs(self):
        return tf.ones(self.postShape, tf.float32)

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.postShape = (config.num_hidden_layers, 2, 1, config.n_head, 1, 64)
        self.postPastWeight = PositionalInvariantAddLayer(self.postShape)

    @tf.function
    def call(self, past):
        return self.postPastWeight(past)

    @tf.function(
        input_signature=[tf.TensorSpec((None, None, 2, 1, None, 1, 64), tf.float32)]
    )
    def serving(self, inputs):
        tf.print(inputs)
        output = self.call(inputs)
        return self.serving_output(output)

    def serving_output(self, outputs):
        return outputs
