import numpy as np

def numpy_conv2d(inputs, kernels, strides, padding, return_kernel=False):
    '''
    Convolute `inputs` with `kernels`.
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of
            channels, height, width)
        kernels: A `numpy.ndarray` of shape (number of kernels, 
            number of channels, height, width). The number of channels 
            of `kernels` must match with that of `inputs`.
        strides: A tuple of two positive `int`.
        padding: A non-negative `int`.
        return_kernel: A `bool`. If `True`, return the unrolled kernel
            and the convoluted outputs, otherwise, return only the 
            convoluted outputs.
        
    Returns:
        outputs: A `numpy.ndarray`. The convoluted outputs.
    '''
    _inputs = _padding(inputs, padding, 'normal')
    outputs_shape = compute_outputs_shape(_inputs, kernels, strides, 'normal')
    _kernels = _unroll_kernel(_inputs, kernels, outputs_shape, 'normal')
    
    _inputs = _flatten(_inputs)
    outputs = np.tensordot(_inputs, _kernels, ((1, 2), (0, 1)))
    outputs = outputs.transpose(0,2,1).reshape(outputs_shape)
    outputs = _stride(outputs, strides, 'normal')
    
    if return_kernel:
        return (outputs, _kernels)
    
    return outputs

def numpy_conv2d_transpose(inputs, kernels, strides, outputs_shape, padding, 
                           output_padding, return_kernel=False):
    '''
    Transposed convolute `inputs` with `kernels`.
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of
            channels, height, width)
        kernels: A `numpy.ndarray` of shape (number of channels, number 
            of kernels, height, width). The number of channels of
            `kernels` must match with that of `inputs`.
        strides: A tuple of two positive `int`.
        outputs_shape: A `numpy.ndarray` of shape (number of samples, 
            number of channels, height, width). It is computed by 
            `compute_outputs_shape()`.
        padding: A non-negative `int`.
        return_kernel: A `bool`. If `True`, return the unrolled kernel
            and the transposed-convoluted outputs, otherwise, return only
            the transposed-convoluted outputs.
        
    Returns:
        outputs: A `numpy.ndarray`. The convoluted outputs.
    '''
    _inputs = _stride(inputs, strides, 'transposed', output_padding)
    _kernels = _unroll_kernel(_inputs, kernels, outputs_shape, 'transposed')

    _inputs = _flatten(_inputs)
    outputs = np.tensordot(_inputs, _kernels, ((1, 2), (0, 1)))
    outputs = outputs.transpose(0,2,1).reshape(outputs_shape)
    outputs = _padding(outputs, padding, 'transposed')
    
    if return_kernel:
        return (outputs, _kernels)
    
    return outputs

#################
## utils
#################

def _padding(inputs, padding, mode):
    '''
    Pad (`mode=='normal'`) or crop (`mode=='transposed'`). 
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of
            channels, height, width).
        padding: A non-negative `int`.
        mode: A `str`. Either 'normal' or 'tranposed'.
        
    Returns:
        outputs: A `numpy.ndarray`. Padded or cropped `inputs`.
    '''
    p = padding
    m, c, h, w = inputs.shape
    
    if mode == 'normal': # do padding
        outputs = np.zeros((m, c, h+2*p, w+2*p), dtype=np.float32)
        outputs[:,:,p:h+p,p:w+p] = inputs
        
    elif mode == 'transposed': # do cropping
        outputs = inputs[:,:,p:h-p,p:w-p]
        
    return outputs

def _flatten(inputs):
    '''
    Flatten the axes for height and width.
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of
            channels, height, width).
        
    Returns:
        outputs: A `numpy.ndarray`. Flattened `inputs`.
    '''
    m, c, h, w = inputs.shape
    return inputs.reshape((m, c, -1))

def _stride(inputs, strides, mode, output_padding=None):
    '''
    If `mode=='normal'`, keep an element of `inputs`every `strides` step
    along both height and width axes. This shrinks `inputs`. If 
    `mode='transposed'`, add `strides` zeros between every element of
    `inputs` along both height and width axes. This expands `inputs`.
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of
            channels, height, width).
        strides: A tuple of two positive `int`.
        output_padding: A non-negative `int` for specifying how many 
            additional rows and columns are added to one side of the 
            output. Only valid when `mode=='transposed'`.
        mode: A `str`. Either 'normal' or 'tranposed'.
        
    Returns:
        outputs: A `numpy.ndarray`. Expanded or shrinked.
    '''
    sh, sw = strides
    m, c, h, w = inputs.shape
    
    if mode == 'normal': # get every other N rows/columns
        outputs = inputs[:,:,::sh,::sw]
        
    elif mode == 'transposed': # add N rows/columns of 0 between two rows/columns
        _h = h + (h - 1) * (sh - 1) + output_padding
        _w = w + (w - 1) * (sw - 1) + output_padding
        outputs = np.zeros((m,c,_h,_w), dtype=np.float32)
        outputs[:,:,::sh,::sw] = inputs
        
    return outputs

def _unroll_kernel(inputs, kernels, outputs_shape, mode):
    '''
    Unroll `kernels`
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of
            channels, height, width)
        kernels: A `numpy.ndarray`. If `mode=='normal'`, it has the 
            shape (number of kernels, number of channels, height, 
            width). If `mode=='transposed'`, it has the shape of (
            number of channels, number of kernels, height, width). The 
            number of channels of `kernels` must match with that of 
            `inputs`.
        outputs_shape: A `numpy.ndarray` of shape (number of samples, 
            number of channels, height, width). It is computed by 
            `compute_outputs_shape()`
        mode: A `str`. Either 'normal' or 'tranposed'
        
    Returns:
        outputs: A `numpy.ndarray`. Unrolled kernel.
    '''
#     __, __, oh, ow = outputs_shape
#     __, ic, ih, iw = inputs.shape
#     kn, kc, kh, kw = kernels.shape
    
#     if mode == 'normal':
#         assert ic == kc
#         _kernel = np.zeros((kc, ih, iw, kn), dtype=np.float32)
#         _kernel[:,:kh,:kw,:] = kernels.transpose(1,2,3,0)
#         _kernel = _kernel.reshape(kc, -1, kn)
#         outputs = np.stack([
#             np.roll(_kernel, iw * x + y, axis=1) for x in range(oh) for y in range(ow)
#         ], axis=2)
        
#     elif mode == 'transposed':
#         assert ic == kn
#         _kernel = np.zeros((kn, oh, ow, kc), dtype=np.float32)
#         _kernel[:,:kh,:kw,:] = kernels.transpose(0,2,3,1)
#         _kernel = _kernel.reshape(kn, -1, kc)
#         outputs = np.stack([
#             np.roll(_kernel, ow * x + y, axis=1) for x in range(ih) for y in range(iw)
#         ], axis=1)
    
    inputs_shape = inputs.shape

    if mode == 'transposed':
        inputs_shape, outputs_shape = outputs_shape, inputs_shape
        kernels = kernels.transpose(1,0,2,3)    

    __, ic, ih, iw = inputs_shape
    __, oc, oh, ow = outputs_shape
    kn, kc, kh, kw = kernels.shape
    
    _kernel = np.zeros((kc, ih, iw, kn), dtype=np.float32)
    _kernel[:,:kh,:kw,:] = kernels.transpose(1,2,3,0)
    _kernel = _kernel.reshape(kc, -1, kn)

    r = np.arange(oh*ow) // ow * (kw - 1)
    y, x = np.ogrid[:ih*iw,:oh*ow]
    idx = (y - (x + r)) % (ih*iw)

    if mode == 'transposed':
        idx = idx.T

    outputs = _kernel[:,idx,:]
        
    return outputs

def compute_outputs_shape(inputs, kernels, strides, mode, output_padding=None):
    '''
    Compute the output shape of the normal/transposed convolution.
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of
            channels, height, width)
        kernels: A `numpy.ndarray`. If `mode=='normal'`, it has the 
            shape (number of kernels, number of channels, height, 
            width). If `mode=='transposed'`, it has the shape of (
            number of channels, number of kernels, height, width). The 
            number of channels of `kernels` must match with that of 
            `inputs`.
        strides: A tuple of two positive `int`.
        mode: A `str`. Either 'normal' or 'tranposed'
        output_padding: A non-negative `int` for specifying how many 
            additional rows and columns are added to one side of the 
            output. Only valid when `mode=='transposed'`.
        
    Returns:
        outputs: A `numpy.ndarray`. Output shape.
    '''
    sh, sw = strides
    im, ic, ih, iw = inputs.shape
    kn, kc, kh, kw = kernels.shape
    
    if mode == 'normal':
        outputs_shape = (
            im, 
            kn,
            ih - kh + 1, 
            iw - kw + 1)
    
    elif mode == 'transposed':
        outputs_shape = (
            im, 
            kc,
            (ih - 1) * sh + kh + output_padding,
            (iw - 1) * sw + kw + output_padding)

    return outputs_shape