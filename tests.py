import torch
import numpy as np
import tensorflow as tf
from conv2d_transpose_numpy import numpy_conv2d, numpy_conv2d_transpose, \
                                   compute_outputs_shape

def test_normal_convolution(inputs, kernels, strides, print_arrays=False):
    '''
    Test convolution, and compare results with Tensorflow's and Torch's 
    implementation.
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of
            channels, height, width)
        kernels: A `numpy.ndarray` of shape (number of kernels, number
            of channels, height, width). The number of channels of
            `kernels` must match with that of `inputs`.
        strides: A positive `int` 
        print_arrays: A `bool`. If `True`, print the convoluted arrays
            by implementation
    '''
    
    assert inputs.shape[1] == kernels.shape[1]
    
    print('Normal Convolution tests\n')
    
    # Tensorflow
    inputs_tf = inputs.transpose(0,2,3,1)
    kernels_tf = kernels.transpose(2,3,1,0)
    outputs_tensorflow = tf.nn.conv2d(
        inputs_tf, kernels_tf, strides=strides, 
        padding='VALID')
    outputs_tensorflow = outputs_tensorflow.numpy()
    outputs_tensorflow = outputs_tensorflow.transpose(0,3,1,2)

    # Torch
    inputs_torch = torch.from_numpy(inputs)
    kernels_torch = torch.from_numpy(kernels)
    outputs_torch = torch.nn.functional.conv2d(
        inputs_torch, kernels_torch, stride=strides, padding='valid')
    outputs_torch = outputs_torch.numpy()

    # Numpy
    outputs_numpy = numpy_conv2d(inputs, kernels, strides, padding=0)
    
    print('AllClose Check: TensorFlow vs Numpy:', 
          'Passed' if np.allclose(outputs_tensorflow, outputs_numpy)\
              else 'Failed')
    
    print('AllClose Check: Torch vs Numpy:',
          'Passed' if np.allclose(outputs_torch, outputs_numpy)\
              else 'Failed')
    
    print('\nOutput Shapes:')
    print('Numpy:', outputs_numpy.shape)
    print('PyTorch:', outputs_torch.shape)
    print('Tensorflow:', outputs_tensorflow.shape)
    
    if print_arrays:
        print('\n\nTensorflow')
        print(outputs_tensorflow)
        print('\n\nTorch')
        print(outputs_torch)
        print('\n\nNumpy')
        print(outputs_numpy)
        
def test_transposed_convolution(inputs, kernels, strides, padding, 
                                print_arrays=False):
    '''
    Test Transposed Convolution, and compare results with Tensorflow's
    and Torch's implementation.
    
    Args:
        inputs: A `numpy.ndarray` of shape (number of samples, number of 
            channels, height, width)
        kernels: A `numpy.ndarray` of shape (number of channels, number 
            of kernels, height, width). The number of channels of
            `kernels` must match with that of `inputs`.
        strides: A positive `int` 
        padding: A non-negative `int` for Torch and Numpy 
            implementations only. Tensorflow does not use padding
        print_arrays: A `bool`. If `True`, print the convolution-
            transposed arrays by implementation
    '''
    
    assert inputs.shape[1] == kernels.shape[0]
    
    print('Transposed Convolution tests\n')
    
    outputs_shape = compute_outputs_shape(inputs, kernels, strides, 
                                         mode='transposed')
    
    # Tensorflow
    inputs_tf = inputs.transpose(0,2,3,1)
    kernels_tf = kernels.transpose(2,3,1,0)
    outputs_shape_tf = (
        outputs_shape[0], outputs_shape[2], 
        outputs_shape[3], outputs_shape[1])
    
    # outputs_tensorflow = tf.nn.conv2d_transpose(
    #     inputs_tf, kernels_tf, strides=strides, output_shape=outputs_shape_tf, 
    #     padding='VALID')
    # outputs_tensorflow = outputs_tensorflow.numpy()
    # outputs_tensorflow = outputs_tensorflow.transpose(0,3,1,2)
    
    kwargs = dict(
        filters=kernels_tf.shape[2],
        kernel_size=kernels_tf.shape[:2],
        strides=strides,
        use_bias=False,
    )
    
    outputs_tensorflow1 = tf.keras.layers.Conv2DTranspose(
        padding='valid',
        output_padding=None,
        **kwargs
    )(inputs_tf).numpy().transpose(0,3,1,2)
    
    outputs_tensorflow2 = tf.keras.layers.Conv2DTranspose(
        padding='same',
        output_padding=None,
        **kwargs
    )(inputs_tf).numpy().transpose(0,3,1,2)
    
    outputs_tensorflow3 = tf.keras.layers.Conv2DTranspose(
        padding='valid',
        output_padding=0,
        **kwargs
    )(inputs_tf).numpy().transpose(0,3,1,2)
    
    outputs_tensorflow4 = tf.keras.layers.Conv2DTranspose(
        padding='same',
        output_padding=0,
        **kwargs
    )(inputs_tf).numpy().transpose(0,3,1,2)
    
    # Torch
    inputs_torch = torch.from_numpy(inputs)
    kernels_torch = torch.from_numpy(kernels)
    conv_transposed2d_obj = torch.nn.ConvTranspose2d(
        in_channels=kernels.shape[0], 
        out_channels=kernels.shape[1], 
        kernel_size=kernels.shape[:2], 
        stride=strides, 
        padding=padding, 
        output_padding=0, 
        bias=False
    )
    conv_transposed2d_obj.weight = torch.nn.Parameter(kernels_torch)
    outputs_torch = conv_transposed2d_obj(inputs_torch)
    outputs_torch = outputs_torch.detach().numpy()

    # Numpy
    outputs_numpy = numpy_conv2d_transpose(
        inputs, kernels, strides, outputs_shape, padding)
    
    print('AllClose Check: Torch vs Numpy:',
          'Passed' if np.allclose(outputs_torch, outputs_numpy)\
              else 'Failed')
    
    print('\nOutput Shapes:')
    print('Numpy:', outputs_numpy.shape)
    print('PyTorch:', outputs_torch.shape)
    
    print('\nVarious Tensorflow Output Shapes:')
    print('padding=\'valid\' output_padding=None', outputs_tensorflow1.shape)
    print('padding=\'same\'  output_padding=None', outputs_tensorflow2.shape)
    print('padding=\'valid\' output_padding=0   ', outputs_tensorflow3.shape)
    print('padding=\'same\'  output_padding=0   ', outputs_tensorflow4.shape)
    
    if print_arrays:
        print('\n\nTorch')
        print(outputs_torch)
        print('\n\nNumpy')
        print(outputs_numpy)
        print('\n\nTensorflow: padding=\'valid\' output_padding=None')
        print(outputs_tensorflow1)
        print('\n\nTensorflow: padding=\'same\'  output_padding=None')
        print(outputs_tensorflow2)
        print('\n\nTensorflow: padding=\'valid\' output_padding=0   ')
        print(outputs_tensorflow3)
        print('\n\nTensorflow: padding=\'same\'  output_padding=0   ')
        print(outputs_tensorflow4)
