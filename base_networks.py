#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abhijit
Contains a few Famous CNN architectures which can be used as shared layers in Siamese or Triplet networks
"""

from abc import ABC, abstractmethod

from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D, BatchNormalization, ZeroPadding2D
import keras.backend as K
import warnings


class BaseNetwork(ABC):
    """Base class for Various Famous architectures
    All the classes inheriting this base class must implement the get_network(self) method
    """

    def __init__(self, input_tensor=None, input_shape=None):
        """ input_tensor or input_shape """
        if K.image_data_format() == 'channels_first':
            raise ValueError("Check Keras backend, The backend must be Tensorflow")
        if input_tensor is None:
            self.img_input = Input(shape=input_shape)
            warnings.warn("Using only input_shape argument might cause problems while using in shared networks, \
             Use input_tensor for shared networks", RuntimeWarning)
        else:
            if not K.is_keras_tensor(input_tensor):
                self.img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                self.img_input = input_tensor

        super().__init__()

    @abstractmethod
    def network(self):
        """Contains the network definition
        """
        pass


class InceptionV3(BaseNetwork):
    """Instantiates the Inception V3 architecture and returns it

    Keyword Arguments:
    input_tensor --Tensor object Default None
    input_shape --Tuple: The image input shape example (299,299,3) default None
    pooling --str: Pooling operation to apply to the end of the network
                   Default = None ('avg','max' are valid)

    output_layers --str: Model is constructed until that layer (Default: mixed10)
    """

    def __init__(self, input_tensor=None, input_shape=None, pooling=None, output_layer='mixed10'):

        all_output_layers = ['mixed{}'.format(i) for i in range(11)]
        assert output_layer in all_output_layers, "Output layer must be one of {}".format(all_output_layers)

        self.output_layer = output_layer
        assert pooling in [None, 'avg', 'max'], "Pooling must be average or Max or None"
        self.pooling = pooling
        super().__init__(input_tensor, input_shape)

    @staticmethod
    def _conv2d_bn(x,
                   filters,
                   num_row,
                   num_col,
                   padding='same',
                   strides=(1, 1),
                   name=None):
        """Utility function to apply conv + BN.
        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.
        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x

    @property
    def network(self):
        """Call this method to instantiate the Inceptionv3 architecture
        """
        x = self._get_network()
        if self.pooling:
            if self.pooling == "avg":
                x = GlobalAveragePooling2D(name="global_avg_pooling")(x)
            else:
                x = GlobalMaxPool2D(name="Global_max_pooling")(x)
        return x

    def _get_network(self):
        """Inceptionv3 architecture from keras
        """
        channel_axis = 3
        x = self._conv2d_bn(self.img_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = self._conv2d_bn(x, 32, 3, 3, padding='valid')
        x = self._conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self._conv2d_bn(x, 80, 1, 1, padding='valid')
        x = self._conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = self._conv2d_bn(x, 64, 1, 1)

        branch5x5 = self._conv2d_bn(x, 48, 1, 1)
        branch5x5 = self._conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self._conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        if self.output_layer == 'mixed0':
            return x

        # mixed 1: 35 x 35 x 256
        branch1x1 = self._conv2d_bn(x, 64, 1, 1)

        branch5x5 = self._conv2d_bn(x, 48, 1, 1)
        branch5x5 = self._conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self._conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        if self.output_layer == 'mixed1':
            return x

        # mixed 2: 35 x 35 x 256
        branch1x1 = self._conv2d_bn(x, 64, 1, 1)

        branch5x5 = self._conv2d_bn(x, 48, 1, 1)
        branch5x5 = self._conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self._conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        if self.output_layer == 'mixed2':
            return x

        # mixed 3: 17 x 17 x 768
        branch3x3 = self._conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = self._conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self._conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self._conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

        if self.output_layer == 'mixed3':
            return x

        # mixed 4: 17 x 17 x 768
        branch1x1 = self._conv2d_bn(x, 192, 1, 1)

        branch7x7 = self._conv2d_bn(x, 128, 1, 1)
        branch7x7 = self._conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = self._conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self._conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

        if self.output_layer == 'mixed4':
            return x

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self._conv2d_bn(x, 192, 1, 1)

            branch7x7 = self._conv2d_bn(x, 160, 1, 1)
            branch7x7 = self._conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = self._conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self._conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = self._conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self._conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = self._conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self._conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(5 + i))

            if self.output_layer == 'mixed5':
                return x

        if self.output_layer == 'mixed6':
            return x

        # mixed 7: 17 x 17 x 768
        branch1x1 = self._conv2d_bn(x, 192, 1, 1)

        branch7x7 = self._conv2d_bn(x, 192, 1, 1)
        branch7x7 = self._conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = self._conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self._conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self._conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self._conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        if self.output_layer == 'mixed7':
            return x

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self._conv2d_bn(x, 192, 1, 1)
        branch3x3 = self._conv2d_bn(branch3x3, 320, 3, 3,
                                    strides=(2, 2), padding='valid')

        branch7x7x3 = self._conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = self._conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = self._conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = self._conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

        if self.output_layer == 'mixed8':
            return x

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self._conv2d_bn(x, 320, 1, 1)

            branch3x3 = self._conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = self._conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = self._conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

            branch3x3dbl = self._conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = self._conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = self._conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = self._conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self._conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(9 + i))

            if self.output_layer == 'mixed9':
                return x

        return x


class VGG16(BaseNetwork):
    """ Instantiates the VGG16 architecture and returns it

    Keyword Arguments:
    input_tensor --Tensor object Default None
    input_shape --Tuple: The image input shape example (299,299,3) default None

    output_layers --str: Model is constructed until that layer (Default:block5_conv3)

    """

    def __init__(self, input_tensor=None, input_shape=None, output_layer='block5_conv3'):
        all_output_layers = ['block{}_pool'.format(i+1) for i in range(5)]
        all_output_layers.append('block5_conv3')

        assert output_layer in all_output_layers, "output_layer must be one of {}".format(all_output_layers)
        self.output_layer = output_layer
        super().__init__(input_tensor, input_shape)

    @property
    def network(self):
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(self.img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        if self.output_layer == 'block1_pool':
            return x

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        if self.output_layer == 'block2_pool':
            return x

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        if self.output_layer == 'block3_pool':
            return x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        if self.output_layer == 'block4_pool':
            return x

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

        if self.output_layer == 'block5_conv3':
            return x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if self.output_layer == 'block5_pool':
            return x


class VGG19(BaseNetwork):
    """ Instantiates The VGG19 architecture and returns it

    Keyword Arguments:
    input_tensor --Tensor object Default None
    input_shape --Tuple: The image input shape example (299,299,3) default None
    output_layers --str: Model is constructed until that layer (Default:block5_conv4)

    """

    def __init__(self, input_tensor=None, input_shape=None, output_layer='block5_conv4'):
        all_output_layers = ['block{}_pool'.format(i+1) for i in range(5)]
        all_output_layers.append('block5_conv4')

        assert output_layer in all_output_layers, "output_layer must be one of {}".format(output_layer)
        self.output_layer = output_layer
        super().__init__(input_tensor, input_shape)

    @property
    def network(self):
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(self.img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        if self.output_layer == 'block1_pool':
            return x

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        if self.output_layer == 'block2_pool':
            return x

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        if self.output_layer == 'block3_pool':
            return x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        if self.output_layer == 'block4_pool':
            return x

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

        if self.output_layer == 'block5_conv4':
            return x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if self.output_layer == 'block5_pool':
            return x


class Resnet50(BaseNetwork):
    """Instantiates a Resnet50 architecture and returns it

    Keyword Arguments:
    input_tensor --Tensor object Default None
    input_shape --Tuple: The image input shape example (299,299,3) default None
    pooling --str: Pooling operation to apply to the end of the network
                   Default = None ('avg','max' are valid)

    """

    def __init__(self, input_tensor=None, input_shape=None, pooling=None):
        assert pooling in [None, 'avg', 'max'], "Pooling must be average or Max or None"
        self.pooling = pooling
        super().__init__(input_tensor, input_shape)

    @staticmethod
    def _identity_block(input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    @staticmethod
    def _conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """A block that has a conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.
        # Returns
            Output tensor for the block.
        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    @property
    def network(self):
        bn_axis = 3
        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(self.img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self._conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self._conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self._conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self._conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        if self.pooling:
            if self.pooling == "avg":
                x = GlobalAveragePooling2D(name="global_avg_pooling")(x)
            else:
                x = GlobalMaxPool2D(name="Global_max_pooling")(x)

        return x
