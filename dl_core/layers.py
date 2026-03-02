from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    x_reshaped = x.reshape(x.shape[0], -1)
    # print(f"{x_reshaped.shape=}, {w.shape=}")

    out = x_reshaped @ w + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = (dout @ w.T).reshape(*x.shape)
    x = x.reshape(x.shape[0], -1)
    dw = (x.T @ dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = np.zeros_like(x)
    dx[x>0] = dout[x>0]

    # dx = dout * (x>0)
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    # forward pass
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(x_shifted)
    p /= p.sum(axis=1, keepdims=True)
    logp = np.log(p + 1e-12)
    loss = -np.mean(logp[np.arange(x.shape[0]), y])

    # gradient
    p[np.arange(x.shape[0]), y] -= 1
    dx = p / x.shape[0]
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0, keepdims=True)
        sample_var = np.var(x, axis=0, keepdims=True)

        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = (x_norm * gamma) + beta
        cache = (sample_mean, sample_var, gamma, beta, eps, x_norm, x)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == "test":
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = (x_norm * gamma) + beta

        cache = (running_mean, running_var, gamma, beta, eps, x_norm, x)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    mean, var, gamma, beta, eps, x_norm, x = cache
    N, D = dout.shape

    # 2. Recompute the missing intermediate variables
    xhat = x_norm              
    xmu = x - mean             # compute the centered input
    std = np.sqrt(var + eps)   # compute standard deviation
    inv_std = 1.0 / std        # compute inverse standard deviation
    
    # Backprop through Scale and Shift (gamma and beta)
    dbeta = np.sum(dout, axis=0)
    dy_scaled = dout 
    dgamma = np.sum(dy_scaled * xhat, axis=0)
    dxhat = dy_scaled * gamma

    # Backprop through Normalization
    dinv_std = np.sum(dxhat * xmu, axis=0)
    dxmu1 = dxhat * inv_std 

    # Backprop through Inverse Standard Deviation and Variance
    dstd = dinv_std * (-1.0 / (std ** 2))
    dvar = dstd * 0.5 * (1.0 / std) # using std here instead of recalculating sqrt

    # Backprop through Variance Calculation
    dsq = (1.0 / N) * np.ones((N, D)) * dvar
    dxmu2 = dsq * 2 * xmu

    # Combine Branches for xmu
    dxmu = dxmu1 + dxmu2

    # Backprop through Mean Subtraction
    dx1 = dxmu * 1.0 
    dmu = np.sum(dxmu * -1.0, axis=0)

    # Backprop through Mean Calculation
    dx2 = (1.0 / N) * np.ones((N, D)) * dmu

    # Combine Branches for x
    dx = dx1 + dx2

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    mean , var, gamma, beta, eps, x_norm, x = cache

    N, D = x.shape

    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean), axis=0) * -0.5 * (var + eps)**-1.5
    dmean = (np.sum(dx_norm, axis=0) * -1 / np.sqrt(var + eps)) + (dvar * np.mean(-2 * (x - mean), axis=0))
    dx = (dx_norm * 1/np.sqrt(var + eps)) + (2 * dvar * (x - mean)/N) + (dmean / N)
    dgamma = np.sum(x_norm * dout, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    x = x.T

    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)
    x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)

    x = x.T
    x_norm = x_norm.T

    out = (x_norm * gamma) + beta

    cache = (sample_mean, sample_var, gamma, beta, eps, x_norm, x)
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    mean, var, gamma, beta, eps, x_norm, x = cache

    x = x.T
    x_norm = x_norm.T
    dout = dout.T

    gamma = gamma.reshape(gamma.shape[0], 1)
    beta = beta.reshape(beta.shape[0], 1)

    N, D = x.shape

    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean), axis=0) * -0.5 * (var + eps)**-1.5
    dmean = (np.sum(dx_norm, axis=0) * -1 / np.sqrt(var + eps)) + (dvar * np.mean(-2 * (x - mean), axis=0))
    dx = (dx_norm * 1/np.sqrt(var + eps)) + (2 * dvar * (x - mean)/N) + (dmean / N)

    x = x.T
    x_norm = x_norm.T
    dout = dout.T
    dx = dx.T

    dgamma = np.sum(x_norm * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    out = None
    mask = None
    
    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x
    else:
        raise NotImplementedError

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    pad = conv_param["pad"]
    stride = conv_param["stride"]

    x_paded = np.pad(x, ((0,), (0,), (pad,), (pad,)))

    H_out = 1 + ((H + 2 * pad - HH) // stride)
    W_out = 1 + ((W + 2 * pad - WW) // stride)

    out = np.zeros((N, F, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride

            x_stride = x_paded[:, : , h_start:h_start+HH, w_start:w_start+WW]
            out[:, :, i, j] = np.sum(x_stride[:, np.newaxis, :, :, :] * w, 
                                                         axis=(2, 3, 4))
            
    out += b[np.newaxis, :, np.newaxis, np.newaxis]
    # N, 1, C, W, H
    # F, C, W, H  
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    pad = conv_param["pad"]
    stride = conv_param["stride"]

    x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)))
    dx_padded = np.zeros_like(x_padded)

    H_out = 1 + ((H + 2 * pad - HH) // stride)
    W_out = 1 + ((W + 2 * pad - WW) // stride)

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    for i in range(H_out):
        for j in range(W_out):
           h_start = i * stride
           w_start = j * stride

           dx_padded[:, :, h_start:h_start+HH, w_start:w_start+WW] += np.sum(
               dout[:, :, i, j, None, None, None] * w,
               axis = 1
           )
           dw += np.sum(dout[:, :, i, j, None, None, None] * x_padded[:, None, :, h_start:h_start+HH, w_start:w_start+WW],
                        axis=0)

    dx = dx_padded[:, :, pad:-pad, pad:-pad] if pad != 0 else dx
    db = np.sum(dout, axis=(0, 2, 3))

    # dout =     (N, F, 1, 1, 1)
    # x_stride = (N, 1, C, HH, WW)
    # w = (F, C, HH, WW)
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    HH = pool_param["pool_height"]
    WW = pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape

    H_out = 1 + (H - HH) // stride
    W_out = 1 + (W - WW) // stride

    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride

            out[:, :, i, j] = np.max(x[:, :, h_start:h_start+HH, w_start:w_start+WW], axis=(2,3))
          
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache

    HH = pool_param["pool_height"]
    WW = pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape

    H_out = 1 + (H - HH) // stride
    W_out = 1 + (W - WW) // stride

    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride

            dx[:, :, h_start:h_start+HH, w_start:w_start+WW] = dout[:, :, i, j, None, None] * (
                (x[:, :, h_start:h_start+HH, w_start:w_start+WW]) == np.max(x[:, :, h_start:h_start+HH, w_start:w_start+WW], 
                                                                            axis=(2, 3), keepdims=True))

    
    # dx = (N, C, HH, WW)

    # dout = (N, C, HH, WW)
    # mask = (N, C,  1,  1)

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    N, C, H, W = x.shape
    x_altered = np.transpose(x, (0, 2, 3, 1)).reshape(-1, C)
    out, cache = batchnorm_forward(x_altered, gamma, beta, bn_param)
    out = np.transpose(out.reshape(N, H, W, C), (0, 3, 1, 2))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    dout_altered = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_altered, cache)
    dx = np.transpose(dx.reshape(N, H, W, C), (0, 3, 1, 2))
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)


    N, C, H, W = x.shape
    D = C // G

    x_reshape = np.reshape(x, (N, G, D, H, W))
    x_flat = np.reshape(x_reshape, (N * G, -1))

    x_flat = x_flat.T

    mean = np.mean(x_flat, axis=0)
    var = np.var(x_flat, axis=0)

    x_norm_flat = (x_flat - mean) / np.sqrt(var + eps)
    x_norm = np.reshape(x_norm_flat, (N, C, H, W))
    mean = mean.T
    var = var.T

    out = (x_norm * gamma) + beta
  
    cache = (mean, var, gamma, beta, eps, x_norm, x, G)
    return out, cache


# Not Implemented by me
# dx dont match yet
def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    mean, var, gamma, beta, eps, x_norm, x, G = cache
    N, C, H, W = x.shape
    D_group = C // G
    M = D_group * H * W  # Elements per group (the "N" in your formula)

    # 1. Gradients for scale and shift (4D accumulation)
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    # 2. Reshape and Transpose to (M, N*G)
    # We apply gamma BEFORE flattening to handle the channel-wise scaling
    dx_norm_4d = dout * gamma
    dx_norm_flat = dx_norm_4d.reshape(N, G, D_group, H, W).reshape(N * G, M).T
    x_flat = x.reshape(N, G, D_group, H, W).reshape(N * G, M).T
    
    # 3. Precise Gradient Math
    std_inv = 1.0 / np.sqrt(var + eps) # Shape (N*G,)
    
    # Step 1: Gradient of Variance
    # sum over M features (axis 0)
    dvar = np.sum(dx_norm_flat * (x_flat - mean), axis=0) * -0.5 * (std_inv**3)
    
    # Step 2: Gradient of Mean
    # This must account for the direct path and the path through variance
    dmean = np.sum(dx_norm_flat * -std_inv, axis=0) + dvar * np.mean(-2.0 * (x_flat - mean), axis=0)
    
    # Step 3: Final dx
    # Standard formula: dx_norm * (1/std) + dvar * (2(x-mu)/M) + dmean/M
    dx_flat = (dx_norm_flat * std_inv) + (dvar * 2.0 * (x_flat - mean) / M) + (dmean / M)

    # 4. Reshape back to (N, C, H, W)
    dx = dx_flat.T.reshape(N, C, H, W)

    return dx, dgamma, dbeta
    return dx, dgamma, dbeta
