from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, \
                         Conv2DTranspose, add, concatenate, AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from loss import *
from backbones import get_backbone


####### custom loss #######
def bg_loss(y_true, y_pred, channel=0):
    # for bg_mask: use dice_loss
    bg_loss_ = 1 - dice_n(y_true, y_pred, channel)
    return bg_loss_


def line_loss(y_true, y_pred, channelLst=[i for i in range(1,9)]):
    # for line_mask: use dice_bce_focal_loss
    alpha = 10
    beta = 100
    line_loss_ = 0
    for i in channelLst:
        dice = dice_n(y_true, y_pred, i)
        bce = reweighting_bce_n(y_true, y_pred, i)
        focal = focal_loss_n(y_true, y_pred, i)
        line_loss_ += 1 - dice + alpha * bce + beta * focal
    return line_loss_


def cspine_loss(y_true, y_pred, channel=9):
    # for cpsine_mask: use dice_focal_loss
    alpha = 100
    cspine_loss_ = 1 - dice_n(y_true, y_pred, channel) + alpha * focal_loss_n(y_true, y_pred, channel)
    return cspine_loss_


def mixed_loss(y_true, y_pred):
    return bg_loss(y_true, y_pred, 0) + 2*line_loss(y_true, y_pred, [i for i in range(1,9)]) + 4*cspine_loss(y_true, y_pred, 9)


def test_loss(y_true, y_pred):
    return border_dice_n(y_true, y_pred, 0)


####### custom metric #######
def metric_dice_1(y_true, y_pred):
    return dice_n(y_true, y_pred, 0)
def metric_dice_2(y_true, y_pred):
    return dice_n(y_true, y_pred, 2)
def metric_dice_3(y_true, y_pred):
    return dice_n(y_true, y_pred, 8)
def metric_dice_4(y_true, y_pred):
    return dice_n(y_true, y_pred, 9)


####### custom model #######
def unet(backbone_name='resnet50', input_shape=(256,256,1), output_channels=1, stage=5):
    # backboned encoder
    backbone, encoder_features = get_backbone(backbone_name, input_shape)
    inpt = backbone.input
    x = backbone.output

    # remove average pooling layer at the end of backbone (for resnet models of certain version of keras)
    if isinstance(backbone.layers[-1], AveragePooling2D):
        x = backbone.get_layer(index=-2).output

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], MaxPooling2D):
        x = Conv3x3BnReLU(x, 512)
        x = Conv3x3BnReLU(x, 512)

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in encoder_features])

    # building decoder blocks
    decoder_filters=(256, 128, 64, 32, 16)
    # decoder_filters=(2048, 1024, 512, 256, 64)
    for i in range(stage):     # [0,1,2,3,4]
        if i < len(skips):
            skip = skips[i]
        else:
            skip = None
        x = decoder_block_deconv(x, skip, decoder_filters[i])

    # model head
    x = Conv2D(output_channels, kernel_size=3, padding='same', activation='sigmoid')(x)

    model = Model(inpt, x)

    sgd = SGD(lr=1e-4, momentum=0.97, decay=1e-6, nesterov=True)
    # metric_lst = [metric_dice_1, metric_dice_2, metric_dice_3, metric_dice_4] + [dice_loss, bg_loss, line_loss, cspine_loss]
    model.compile(sgd, loss=test_loss, metrics=[metric_dice_1])

    return model


def Conv3x3BnReLU(x, n_filters, padding='same', strides=1, activation='relu'):
    x = Conv2D(n_filters, kernel_size=3, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def Conv1x1BnReLU(x, n_filters, padding='same', strides=1, activation='relu'):
    x = Conv2D(n_filters, kernel_size=1, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def decoder_block_up(x, shortcut, n_filters):
    # upsampling
    input_filters = x.shape.as_list()[-1]
    output_filters = shortcut.shape.as_list()[-1] if shortcut is not None else n_filters
    x = Conv1x1BnReLU(x, input_filters//4, padding='same', strides=1, activation='relu')
    x = UpSampling2D()(x)
    x = Conv3x3BnReLU(x, input_filters//4, padding='same', strides=1, activation='relu')
    x = Conv1x1BnReLU(x, output_filters, padding='same', strides=1, activation='relu')
    # add or concatenate
    if shortcut is not None:
        x = add([x, shortcut])
    return x


def decoder_block_deconv(x, shortcut, n_filters):
    # convTranspose
    input_filters = x.shape.as_list()[-1]
    output_filters = shortcut.shape.as_list()[-1] if shortcut is not None else n_filters
    x = Conv1x1BnReLU(x, input_filters//4, padding='same', strides=1, activation='relu')
    x = Conv2DTranspose(input_filters//4, kernel_size=4, padding='same', strides=2)(x)
    x = Conv1x1BnReLU(x, output_filters, padding='same', strides=1, activation='relu')
    # add or concatenate
    if shortcut is not None:
        x = add([x, shortcut])
    return x


if __name__ == '__main__':
    model = unet('orig_unet', input_shape=(512,512,2), output_channels=4, stage=5)
    # model = unet('vgg16', input_shape=(256,256,3), output_channels=1, stage=5)
    model.summary()



