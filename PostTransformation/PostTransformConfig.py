import transformers


class PostTransformConfig(transformers.PretrainedConfig):
    model_type = "PostTransformation"

    def __init__(self, num_hidden_layers=24, n_head=12, **kwargs):
        self.n_head = n_head
        self.num_hidden_layers = num_hidden_layers
        super().__init__(**kwargs)


# def postTransformationFromGPTConfig(config):
#     postConfig = PostTransformConfig.PostTransformConfig(config.num_hidden_layers, config.n_head)
#     return postConfig