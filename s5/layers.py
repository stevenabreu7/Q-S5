from jax import lax
from jax.nn import initializers
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype
from flax.linen.module import Module, compact
from flax.linen.normalization import _canonicalize_axes, _abs_sq
from flax.typing import Array, PRNGKey, Dtype, Shape, Axes
from typing import Any, Callable, Optional, Tuple
import aqt.jax.v2.flax.aqt_flax as aqt
import jax
import jax.numpy as jnp
from .utils.quantization import q_dot_maybe, q_had_maybe


def q_gelu(precision: int):
    """
        Quantized hard squish function to approximate GeLU.
        Operates purely on integers without needing floating points.
    """
    _q_had = q_had_maybe(precision, precision)

    def _hard_sigmoid(x): # this operates purely on integers!
        return jnp.minimum(jnp.maximum(0,x+2), 4) / 4 # jnp.right_shift allows for pure integer input/output!

    def _q_gelu(x):
        return _q_had(x, _hard_sigmoid(x))

    return jax.jit(_q_gelu)


def _compute_stats(x: Array, axes: Axes,
                   dtype: Optional[Dtype],
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - Computes in float32 precision for stability in half precision training.
  - mean and variance are computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.

  Arguments:
    x: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    dtype: Optional dtype specifying the minimal precision. Statistics
      are always at least float32 for stability (default: dtype of x).
    axis_name: Optional name for the pmapped axis to compute mean over.
    axis_index_groups: Optional axis indices.

  Returns:
    A pair ``(mean, var)``.
  """
  if dtype is None:
    dtype = jnp.result_type(x) # TODO is this concerning???
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  dtype = jnp.promote_types(dtype, jnp.float32)
  x = jnp.asarray(x, dtype)

  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(_abs_sq(x), axes)
  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
        lax.pmean(
            concatenated_mean,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - _abs_sq(mean))
  return mean, var

def _q_normalize(mdl: Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Dtype, param_dtype: Dtype,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
               scale_init: Callable[[PRNGKey, Shape, Dtype], Array],
               quantized_hadamard_operator: Callable):
    """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.  
    Arguments:
      mdl: Module to apply the normalization in (normalization params will reside
        in this module).
      x: The input.
      mean: Mean to use for normalization.
      var: Variance to use for normalization.
      reduction_axes: The axes in ``x`` to reduce.
      feature_axes: Axes containing features. A separate bias and scale is learned
        for each specified feature.
      dtype: The dtype of the result (default: infer from input and params).
      param_dtype: The dtype of the parameters.
      epsilon: Normalization epsilon.
      use_bias: If true, add a bias term to the output.
      use_scale: If true, scale the output.
      bias_init: Initialization function for the bias term.
      scale_init: Initialization function for the scaling function. 
    Returns:
      The normalized input.
    """
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    args = [x]
    if use_scale:
        scale = mdl.param('scale', scale_init, reduced_feature_shape,
                            param_dtype).reshape(feature_shape)
        mul *= scale
        args.append(scale)
    y = quantized_hadamard_operator(y, mul)
    if use_bias:
        bias = mdl.param('bias', bias_init, reduced_feature_shape,
                            param_dtype).reshape(feature_shape)
        y += bias
        args.append(bias)
    dtype = canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


class QLayerNorm(Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).

    LayerNorm normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Attributes:
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
      reduction_axes: Axes for computing normalization statistics.
      feature_axes: Feature axes for learned bias and scaling.
    """
    scaling_quantization: int
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = False
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    reduction_axes: Axes = -1
    feature_axes: Axes = -1

    @compact
    def __call__(self, x):
        """Applies layer normalization on the input.

        Args:
          x: the inputs

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        mean, var = _compute_stats(x, self.reduction_axes, self.dtype, None, None)
        scale_q_hadamard = q_had_maybe(self.scaling_quantization, self.scaling_quantization)
        return _q_normalize(
            self, x, mean, var, self.reduction_axes, self.feature_axes,
            self.dtype, self.param_dtype, self.epsilon,
            self.use_bias, self.use_scale,
            self.bias_init, self.scale_init,
            scale_q_hadamard) # TODO is this efficient??


class QSequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            q_bits_aw   (int?, int?): quantization precision for activations and weights
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    q_bits_aw: Tuple[int]
    use_hard_sigmoid: bool = False
    use_q_gelu_approx: bool = False
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    use_qlayernorm_if_quantized: bool = True
    use_layernorm_bias: bool = True

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(step_rescale=self.step_rescale)
        # NOTE: nn.Dense calls dot_general(activation, weights)
        dot = aqt.AqtDotGeneral(q_dot_maybe(*self.q_bits_aw, return_cfg=True))
        act_bits, _ = self.q_bits_aw

        if act_bits is not None and not self.use_hard_sigmoid:
            print("[WARNING] Quantized run with real-valued sigmoid function!")
        if act_bits is not None and not self.use_q_gelu_approx:
            print("[WARNING] Quantized run with real-valued gelu function!")
        if act_bits is not None and self.batchnorm:
            print("[WARNING] Quantized run with real-valued batchnorm!")

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model, dot_general=dot)
            self.out2 = nn.Dense(self.d_model, dot_general=dot)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model, dot_general=dot)

        if self.batchnorm:
            print("using batchnorm")
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            if act_bits is None or not self.use_qlayernorm_if_quantized:
                print(f"using layernorm: bias={self.use_layernorm_bias}")
                self.norm = nn.LayerNorm(use_bias=self.use_layernorm_bias)
            else:
                # only use qlayernorm if activations are quantized
                print(f"using qlayernorm with precision: {act_bits}")
                self.norm = QLayerNorm(scaling_quantization=act_bits)

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

        # multiplicative gating function
        if act_bits is not None:
            self.mult_gate = q_had_maybe(act_bits, act_bits)
        else:
            self.mult_gate = jax.numpy.multiply

        # nonlinear activation function
        self.sigmoid = jax.nn.sigmoid
        if self.use_hard_sigmoid:
            self.sigmoid = jax.nn.hard_sigmoid
        self.gelu = nn.gelu
        if self.use_q_gelu_approx:
            self.gelu = q_gelu(precision=act_bits)

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)

        if self.activation in ["full_glu"]:
            x = self.drop(self.gelu(x))
            x = self.mult_gate(self.out1(x), self.sigmoid(self.out2(x)))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(self.gelu(x))
            x = self.mult_gate(x, self.sigmoid(self.out2(x)))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(self.gelu(x))
            x = self.mult_gate(x, self.sigmoid(self.out2(x1)))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(self.gelu(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x
