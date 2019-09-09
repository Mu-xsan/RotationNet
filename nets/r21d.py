import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
def R2Plus1DNet(inputs, training = False, num_classes = 4, weight_decay = 0.0, layer_sizes=[1,1,1,1]):

    def SpatioTemporalConv(inputs, out_channels, kernel_size, padding=(0, 0, 0), stride=(1, 1, 1),  bias=False, first_conv=False, training = False):
        _, d, h, w, in_channels = inputs.shape.as_list()
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])
        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/(kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        padlayers = tf.keras.layers.ZeroPadding3D(padding = spatial_padding,data_format = "channels_last")
        out = tf.keras.layers.ZeroPadding3D(padding = spatial_padding,
                                            data_format = "channels_last")(inputs)
        out = tf.keras.layers.Conv3D(filters = intermed_channels,
                                     kernel_size = spatial_kernel_size,
                                     strides = spatial_stride,
                                     padding = "valid",
                                     use_bias = bias,
                                     data_format = "channels_last")(out)
        out = tf.keras.layers.BatchNormalization(axis = 4)(out, training=training)
        out = tf.nn.relu(out)
        out = tf.keras.layers.ZeroPadding3D(padding = temporal_padding,
                                            data_format = "channels_last")(out)
        out = tf.keras.layers.Conv3D(filters = out_channels,
                                     kernel_size = temporal_kernel_size,
                                     strides = temporal_stride,
                                     padding = "valid",
                                     use_bias = bias,
                                     data_format = "channels_last")(out)

        return out
    def SpatioTemporalResBlock(inputs, out_channels, kernel_size, downsample=False, training=False):
        _, d, h, w, in_channels = inputs.shape.as_list()
        #padding = kernel_size // 2
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        stride = (1, 1, 1)
        if downsample:
            stride = (2, 2, 2)
         
        out = SpatioTemporalConv(inputs,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 padding = padding,
                                 stride = stride,
                                 training=training)
        out = tf.keras.layers.BatchNormalization(axis = 4)(out)
        out = tf.nn.relu(out)
        out = SpatioTemporalConv(out,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 padding = padding,
                                 training=training)
        out = tf.keras.layers.BatchNormalization(axis = 4)(out)
        if downsample:
            inputs = SpatioTemporalConv(inputs,
                                        out_channels = out_channels,
                                        kernel_size = (1, 1, 1), 
                                        stride = (2, 2, 2),
                                        training = training)
            inputs = tf.keras.layers.BatchNormalization(axis = 4)(inputs)
        res = tf.nn.relu(inputs + out)
        return res

    def SpatioTemporalResLayer(inputs, out_channels, kernel_size, layer_size,  downsample = False, training=False):
        net = SpatioTemporalResBlock(inputs,
                                     out_channels = out_channels,
                                     kernel_size = kernel_size,
                                     downsample = downsample,
                                     training = training)
        for i in range(layer_size - 1):
            net = SpatioTemporalResBlock(net,
                                         out_channels = out_channels,
                                         kernel_size = kernel_size,
                                         training = training)
        return net                     
        
  
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    initializer = tf.glorot_uniform_initializer()
    with tf.variable_scope('R2Plus1DNet',
                            regularizer = regularizer,
                            #initializer = initializer
                          ):
        #with tf.variable_scope('conv1'):
        net = SpatioTemporalConv(inputs, 
                                 out_channels=64, 
                                 kernel_size=(3,7,7),
                                 padding=(1, 3, 3), 
                                 stride=(1, 2, 2),
                                 training=training)
        net = tf.keras.layers.BatchNormalization(axis = 4)(net, training=training)
        net = tf.nn.relu(net)
        #with tf.variable_scope('conv2'):
        net = SpatioTemporalResLayer(net,
                                     out_channels = 64,
                                     kernel_size = (3, 3, 3),
                                     layer_size = layer_sizes[0],
                                     training = training)
        #with tf.variable_scope('conv3'):
        net = SpatioTemporalResLayer(net,
                                     out_channels = 128,
                                     kernel_size = (3, 3, 3),
                                     layer_size = layer_sizes[1],
                                     downsample=True,
                                     training = training)
        #with tf.variable_scope('conv4'):
        net = SpatioTemporalResLayer(net,
                                     out_channels = 256,
                                     kernel_size = (3, 3, 3),
                                     layer_size = layer_sizes[2],
                                     downsample=True,
                                     training = training)
        #with tf.variable_scope('conv5'):
        net = SpatioTemporalResLayer(net,
                                     out_channels = 512,
                                     kernel_size = (3, 3, 3),
                                     layer_size = layer_sizes[2],
                                     downsample=True,
                                     training = training)
        #with tf.variable_scope('pooling'):
        net = tf.keras.layers.GlobalAveragePooling3D(data_format = "channels_last")(net)
        #with tf.variable_scope('fc'):
        logits = tf.keras.layers.Dense(256, input_dim = 512)(net)
        logits = tf.nn.relu(logits)
        logits = tf.keras.layers.Dense(num_classes,input_dim = 256)(net)
    return net, logits
