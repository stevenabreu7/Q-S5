import aqt.jax.v2.flax.aqt_flax as aqt
import jax
import jax.numpy as np
from flax import linen as nn
from typing import Tuple
from .qlayers import QSequenceLayer
from .utils.quantization import q_dot_maybe


class QStackedEncoderModel(nn.Module):
    """ Defines a stack of S5 layers to be used as an encoder.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                     we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            q_bits_aw   (int?, int?): quantization precision for activations and weights
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_model: int
    n_layers: int
    q_bits_aw: Tuple[int]
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    use_hard_sigmoid: bool = False
    use_q_gelu_approx: bool = False
    use_qlayernorm_if_quantized: bool = True
    use_layernorm_bias: bool = True

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        # NOTE: nn.Dense calls dot_general(activation, weights)
        dot = aqt.AqtDotGeneral(q_dot_maybe(*self.q_bits_aw, return_cfg=True))
        self.encoder = nn.Dense(self.d_model, dot_general=dot)
        self.layers = [
            QSequenceLayer(
                ssm=self.ssm,
                dropout=self.dropout,
                d_model=self.d_model,
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                batchnorm=self.batchnorm,
                bn_momentum=self.bn_momentum,
                step_rescale=self.step_rescale,
                q_bits_aw=self.q_bits_aw,
                use_hard_sigmoid=self.use_hard_sigmoid,
                use_q_gelu_approx=self.use_q_gelu_approx,
                use_qlayernorm_if_quantized=self.use_qlayernorm_if_quantized,
                use_layernorm_bias=self.use_layernorm_bias,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x, integration_timesteps):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x

# TODO: Determine if we need to quantize the sum/mean operation in here...
# If so, we might have to slightly retool this as a higher order function which returns the masked_meanpool w/ proper dot_operators...
def masked_meanpool(x, lengths):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.
    Args:
         x (float32): input sequence (L, d_model)
         lengths (int32):   the original length of the sequence before padding
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    L = x.shape[0]
    mask = np.arange(L) < lengths
    return np.sum(mask[..., None]*x, axis=0)/lengths


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


class QClassificationModel(nn.Module):
    """ S5 classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            q_bits_aw   (int?, int?): quantization precision for activations and weights
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            mode        (str):      Options: [pool: use mean pooling, last: just take
                                                                       the last state]
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    q_bits_aw: Tuple[int]
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    use_hard_sigmoid: bool = False
    use_q_gelu_approx: bool = False
    use_qlayernorm_if_quantized: bool = True
    use_layernorm_bias: bool = True

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.encoder = QStackedEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            batchnorm=self.batchnorm,
                            bn_momentum=self.bn_momentum,
                            step_rescale=self.step_rescale,
                            q_bits_aw=self.q_bits_aw,
                            use_hard_sigmoid=self.use_hard_sigmoid,
                            use_q_gelu_approx=self.use_q_gelu_approx,
                            use_qlayernorm_if_quantized=self.use_qlayernorm_if_quantized,
                            use_layernorm_bias=self.use_layernorm_bias,
                                        )
        # NOTE: nn.Dense calls dot_general(activation, weights)
        dot = aqt.AqtDotGeneral(q_dot_maybe(*self.q_bits_aw, return_cfg=True))
        self.decoder = nn.Dense(self.d_output, dot_general=dot)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = np.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError("Mode must be in ['pool'] for self.padded=True (for now...)")
            else:
                x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


# Here we call vmap to parallelize across a batch of input sequences
QBatchClassificationModel = nn.vmap(
    QClassificationModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')


# For Document matching task (e.g. AAN)
class QRetrievalDecoder(nn.Module):
    """
    Defines the decoder to be used for document matching tasks,
    e.g. the AAN task. This is defined as in the S4 paper where we apply
    an MLP to a set of 4 features. The features are computed as described in
    Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
    Args:
        d_output    (int32):    the output dimension, i.e. the number of classes
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                    we usually refer to this size as H
        q_bits_aw   (int?, int?): quantization precision for activations and weights
    """
    d_model: int
    d_output: int
    q_bits_aw: Tuple[int]

    def setup(self):
        """
        Initializes 2 dense layers to be used for the MLP.
        """
        # NOTE: nn.Dense calls dot_general(activation, weights)
        dot = aqt.AqtDotGeneral(q_dot_maybe(*self.q_bits_aw, return_cfg=True))
        self.layer1 = nn.Dense(self.d_model, dot_general=dot)
        self.layer2 = nn.Dense(self.d_output, dot_general=dot)

    def __call__(self, x):
        """
        Computes the input to be used for the softmax function given a set of
        4 features. Note this function operates directly on the batch size.
        Args:
             x (float32): features (bsz, 4*d_model)
        Returns:
            output (float32): (bsz, d_output)
        """
        x = self.layer1(x)
        x = nn.gelu(x)
        return self.layer2(x)


class QRetrievalModel(nn.Module):
    """ S5 Retrieval classification model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, constructing 4 features which are fed into a MLP,
    and a softmax operation. Note that unlike the standard classification model above,
    the apply function of this model operates directly on the batch of data (instead of calling
    vmap on this model).
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            q_bits_aw   (int?, int?): quantization precision for activations and weights
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    q_bits_aw: Tuple[int]
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    use_hard_sigmoid: bool = False
    use_q_gelu_approx: bool = False
    use_qlayernorm_if_quantized: bool = True
    use_layernorm_bias: bool = True

    def setup(self):
        """
        Initializes the S5 stacked encoder and the retrieval decoder. Note that here we
        vmap over the stacked encoder model to work well with the retrieval decoder that
        operates directly on the batch.
        """
        QBatchEncoderModel = nn.vmap(
            QStackedEncoderModel,
            in_axes=(0, 0),
            out_axes=0,
            variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
            split_rngs={"params": False, "dropout": True}, axis_name='batch'
        )

        self.encoder = QBatchEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            batchnorm=self.batchnorm,
                            bn_momentum=self.bn_momentum,
                            step_rescale=self.step_rescale,
                            q_bits_aw=self.q_bits_aw,
                            use_hard_sigmoid=self.use_hard_sigmoid,
                            use_q_gelu_approx=self.use_q_gelu_approx,
                            use_qlayernorm_if_quantized=self.use_qlayernorm_if_quantized,
                            use_layernorm_bias=self.use_layernorm_bias,
                                        )
        BatchRetrievalDecoder = nn.vmap(
            QRetrievalDecoder,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )

        self.decoder = BatchRetrievalDecoder(
                                d_model=self.d_model,
                                d_output=self.d_output,
                                q_bits_aw=self.q_bits_aw
                                          )

    def __call__(self, input, integration_timesteps):  # input is a tuple of x and lengths
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence. The encoded features are constructed as in
        Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
        Args:
             input (float32, int32): tuple of input sequence and prepadded sequence lengths
                input sequence is of shape (2*bsz, L, d_input) (includes both documents) and
                lengths is (2*bsz,)
        Returns:
            output (float32): (d_output)
        """
        x, lengths = input  # x is 2*bsz*seq_len*in_dim, lengths is: (2*bsz,)
        x = self.encoder(x, integration_timesteps)  # The output is: 2*bszxseq_lenxd_model
        outs = batch_masked_meanpool(x, lengths)  # Avg non-padded values: 2*bszxd_model
        outs0, outs1 = np.split(outs, 2)  # each encoded_i is bszxd_model
        features = np.concatenate([outs0, outs1, outs0-outs1, outs0*outs1], axis=-1)  # bszx4*d_model
        out = self.decoder(features)
        return nn.log_softmax(out, axis=-1)


class QRegressionModel(nn.Module):
    """ S5 classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            q_bits_aw   (int?, int?): quantization precision for activations and weights
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            mode        (str):      Options: [pool: use mean pooling, last: just take
                                                                       the last state]
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    q_bits_aw: Tuple[int]
    padded: bool = False
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    use_hard_sigmoid: bool = False
    use_q_gelu_approx: bool = False
    use_qlayernorm_if_quantized: bool = True
    use_layernorm_bias: bool = True

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.encoder = QStackedEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            batchnorm=self.batchnorm,
                            bn_momentum=self.bn_momentum,
                            step_rescale=self.step_rescale,
                            q_bits_aw=self.q_bits_aw,
                            use_hard_sigmoid=self.use_hard_sigmoid,
                            use_q_gelu_approx=self.use_q_gelu_approx,
                            use_qlayernorm_if_quantized=self.use_qlayernorm_if_quantized,
                            use_layernorm_bias=self.use_layernorm_bias,
                                        )
        # NOTE: nn.Dense calls dot_general(activation, weights)
        dot = aqt.AqtDotGeneral(q_dot_maybe(*self.q_bits_aw, return_cfg=True))
        self.decoder = nn.Dense(self.d_output, dot_general=dot)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder(x, integration_timesteps)
        return self.decoder(x)
        

# Here we call vmap to parallelize across a batch of input sequences
QBatchRegressionModel = nn.vmap(
    QRegressionModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')
