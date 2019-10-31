from ops1 import new_conv_layer, bottleneck_block, bottleneck_block1, avg_pool, flatten_layer, fc_layer, dropout
import tensorflow as tf
from ops1 import max_pool1,new_conv_layer1,avg_pool1
def create_network50(X, h, keep_prob, numClasses):
    num_channels = X.get_shape().as_list()[-1]
    res1 = new_conv_layer1(inputs=X,
                          layer_name='res1',
                          stride=2,
                          num_inChannel=num_channels,
                          filter_size=7,
                          num_filters=64,
                          batch_norm=True,
                          use_relu=True)
    print('---------------------')
    print('X')
    print(X.get_shape())
    print('---------------------')
    print('---------------------')
    print('Before maxpool,Res1')
    print(res1.get_shape())
    print('---------------------')

    res1 = max_pool1(res1, ksize=3, stride=2, name='res1_max_pool')
    print('---------------------')
    print('After maxpool,Res1')
    print(res1.get_shape())
    print('---------------------')
    # Res2
    with tf.variable_scope('Res2'):
        res2a = bottleneck_block(res1, 64, block_name='res2a',
                                 s1=1, k1=1, nf1=64, name1='res2a_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2a_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res2a_branch2c',
                                 s4=1, k4=1, name4='res2a_branch1', first_block=True)
        print('Res2a')
        print(res2a.get_shape())
        print('---------------------')
        res2b = bottleneck_block(res2a, 256, block_name='res2b',
                                 s1=1, k1=1, nf1=64, name1='res2b_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2b_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res2b_branch2c',
                                 s4=1, k4=1, name4='res2b_branch1', first_block=False)
        print('Res2b')
        print(res2b.get_shape())
        print('---------------------')
        res2c = bottleneck_block(res2b, 256, block_name='res2c',
                                 s1=1, k1=1, nf1=64, name1='res2c_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2c_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res2c_branch2c',
                                 s4=1, k4=1, name4='res2c_branch1', first_block=False)
        print('Res2c')
        print(res2c.get_shape())
        print('---------------------')

    # Res3
    with tf.variable_scope('Res3'):
        res3a = bottleneck_block(res2c, 256, block_name='res3a',
                                 s1=2, k1=1, nf1=128, name1='res3a_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3a_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res3a_branch2c',
                                 s4=2, k4=1, name4='res3a_branch1', first_block=True)
        print('Res3a')
        print(res3a.get_shape())
        print('---------------------')
        res3b = bottleneck_block(res3a, 512, block_name='res3b',
                                 s1=1, k1=1, nf1=128, name1='res3b_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3b_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res3b_branch2c',
                                 s4=1, k4=1, name4='res2b_branch1', first_block=False)
        print('Res3b')
        print(res3b.get_shape())
        print('---------------------')
        res3c = bottleneck_block(res3b, 512, block_name='res3c',
                                 s1=1, k1=1, nf1=128, name1='res3c_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3c_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res3c_branch2c',
                                 s4=1, k4=1, name4='res3c_branch1', first_block=False)
        print('Res3c')
        print(res3c.get_shape())
        print('---------------------')
        res3d = bottleneck_block(res3c, 512, block_name='res3d',
                                 s1=1, k1=1, nf1=128, name1='res3d_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3d_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res3d_branch2c',
                                 s4=1, k4=1, name4='res3d_branch1', first_block=False)
        print('Res3d')
        print(res3d.get_shape())
        print('---------------------')
    
    # Res4
    with tf.variable_scope('Res4'):
        res4a = bottleneck_block(res3d, 512, block_name='res4a',
                                 s1=2, k1=1, nf1=256, name1='res4a_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4a_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4a_branch2c',
                                 s4=2, k4=1, name4='res4a_branch1', first_block=True)
        print('---------------------')
        print('Res4a')
        print(res4a.get_shape())
        print('---------------------')
        res4b = bottleneck_block(res4a, 1024, block_name='res4b',
                                 s1=1, k1=1, nf1=256, name1='res4b_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4b_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4b_branch2c',
                                 s4=1, k4=1, name4='res4b_branch1', first_block=False)
        print('Res4b')
        print(res4b.get_shape())
        print('---------------------')
        res4c = bottleneck_block(res4b, 1024, block_name='res4c',
                                 s1=1, k1=1, nf1=256, name1='res4c_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4c_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4c_branch2c',
                                 s4=1, k4=1, name4='res4c_branch1', first_block=False)
        print('Res4c')
        print(res4c.get_shape())
        print('---------------------')
        res4d = bottleneck_block(res4c, 1024, block_name='res4d',
                                 s1=1, k1=1, nf1=256, name1='res4d_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4d_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4d_branch2c',
                                 s4=1, k4=1, name4='res4d_branch1', first_block=False)
        print('Res4d')
        print(res4d.get_shape())
        print('---------------------')
        res4e = bottleneck_block(res4d, 1024, block_name='res4e',
                                 s1=1, k1=1, nf1=256, name1='res4e_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4e_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4e_branch2c',
                                 s4=1, k4=1, name4='res4e_branch1', first_block=False)
        print('Res4e')
        print(res4e.get_shape())
        print('---------------------')
        res4f = bottleneck_block(res4e, 1024, block_name='res4f',
                                 s1=1, k1=1, nf1=256, name1='res4f_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f_branch2c',
                                 s4=1, k4=1, name4='res4f_branch1', first_block=False)
        print('Res4f')
        print(res4f.get_shape())
        print('---------------------')
    
    
    # Res5
    with tf.variable_scope('Res5'):
        res5a = bottleneck_block(res4f, 1024, block_name='res5a',
                                 s1=2, k1=1, nf1=512, name1='res5a_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5a_branch2b',
                                 s3=1, k3=1, nf3=2048, name3='res5a_branch2c',
                                 s4=2, k4=1, name4='res5a_branch1', first_block=True)
        print('---------------------')
        print('Res5a')
        print(res5a.get_shape())
        print('---------------------')
        res5b = bottleneck_block(res5a, 2048, block_name='res5b',
                                 s1=1, k1=1, nf1=512, name1='res5b_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5b_branch2b',
                                 s3=1, k3=1, nf3=2048, name3='res5b_branch2c',
                                 s4=1, k4=1, name4='res5b_branch1', first_block=False)
        print('Res5b')
        print(res5b.get_shape())
        print('---------------------')
        res5c = bottleneck_block(res5b, 2048, block_name='res5c',
                                 s1=1, k1=1, nf1=512, name1='res5c_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5c_branch2b',
                                 s3=1, k3=1, nf3=2048, name3='res5c_branch2c',
                                 s4=1, k4=1, name4='res5c_branch1', first_block=False)
        print('Res5c')
        print(res5c.get_shape())


        res5c = avg_pool1(res5c, ksize=7, stride=1, name='res5_avg_pool')
        print('---------------------')
        print('Res5c after AVG_POOL')
        print(res5c.get_shape())
        print('---------------------')

    net_flatten, _ = flatten_layer(res5c)
    print('---------------------')
    print('Matrix dimension to the first FC layer')
    print(net_flatten.get_shape())
    print('---------------------')
    # net = fc_layer(net_flatten, h, 'FC1', batch_norm=True, add_reg=True, use_relu=True)
    # print('--------FC1-----------')
    # print(net.get_shape())
    # print('---------------------')
    # net = dropout(net, keep_prob)
    net = fc_layer(net_flatten, numClasses, 'FC1', batch_norm=True, add_reg=True, use_relu=False)
    print('--------FC1-----------')
    print(net.get_shape())
    print('---------------------')
    net = dropout(net, keep_prob)
    return net


def create_network18(X, h, keep_prob, numClasses):
    num_channels = X.get_shape().as_list()[-1]
    res1 = new_conv_layer1(inputs=X,
                           layer_name='res1',
                           stride=2,
                           num_inChannel=num_channels,
                           filter_size=7,
                           num_filters=64,
                           batch_norm=True,
                           use_relu=True)
    print('---------------------')
    print('X')
    print(X.get_shape())
    print('---------------------')
    print('---------------------')
    print('Before maxpool,Res1')
    print(res1.get_shape())
    print('---------------------')

    res1 = max_pool1(res1, ksize=3, stride=2, name='res1_max_pool')
    print('---------------------')
    print('After maxpool,Res1')
    print(res1.get_shape())
    print('---------------------')
    # Res2
    with tf.variable_scope('Res2'):
        res2a = bottleneck_block1(res1, 64, block_name='res2a',
                                 s1=1, k1=3, nf1=64, name1='res2a_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2a_branch2b',
                                 s3=1, k3=3, name3='res2a_branch1',first_block=False)

        print('Res2a')
        print(res2a.get_shape())
        print('---------------------')
        res2b = bottleneck_block1(res2a, 64, block_name='res2b',
                                 s1=1, k1=3, nf1=64, name1='res2b_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2b_branch2b',
                                 s3=1, k3=3, name3='res2b_branch1', first_block=False)
        print('Res2b')
        print(res2b.get_shape())
        print('---------------------')

    # Res3
    with tf.variable_scope('Res3'):
        res3a = bottleneck_block1(res2b, 64, block_name='res3a',
                                 s1=2, k1=3, nf1=128, name1='res3a_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3a_branch2b',
                                 s3=2, k3=3, name3='res3a_branch1', first_block=True)
        print('Res3a')
        print(res3a.get_shape())
        print('---------------------')
        res3b = bottleneck_block1(res3a, 128, block_name='res3b',
                                 s1=1, k1=3, nf1=128, name1='res3b_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3b_branch2b',
                                 s3=1, k3=3, name3='res3b_branch1', first_block=False)
        print('Res3b')
        print(res3b.get_shape())
        print('---------------------')


    # Res4
    with tf.variable_scope('Res4'):
        res4a = bottleneck_block1(res3b, 128, block_name='res4a',
                                 s1=2, k1=3, nf1=256, name1='res4a_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4a_branch2b',
                                 s3=2, k3=3, name3='res4a_branch1', first_block=True)
        print('---------------------')
        print('Res4a')
        print(res4a.get_shape())
        print('---------------------')
        res4b = bottleneck_block1(res4a, 256, block_name='res4b',
                                 s1=1, k1=3, nf1=256, name1='res4b_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4b_branch2b',
                                 s3=1, k3=3, name3='res4b_branch1', first_block=False)
        print('Res4b')
        print(res4b.get_shape())
        print('---------------------')

    # Res5
    with tf.variable_scope('Res5'):
        res5a = bottleneck_block1(res4b, 256, block_name='res5a',
                                 s1=2, k1=3, nf1=512, name1='res5a_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5a_branch2b',
                                 s3=2, k3=3, name3='res5a_branch1', first_block=True)
        print('---------------------')
        print('Res5a')
        print(res5a.get_shape())
        print('---------------------')
        res5b = bottleneck_block1(res5a, 512, block_name='res5b',
                                 s1=1, k1=3, nf1=512, name1='res5b_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5b_branch2b',
                                 s3=1, k3=3, name3='res5b_branch1', first_block=False)
        print('Res5b')
        print(res5b.get_shape())
        print('---------------------')


        res5c = avg_pool1(res5b, ksize=7, stride=1, name='res5_avg_pool')
        print('---------------------')
        print('Res5b after AVG_POOL')
        print(res5c.get_shape())
        print('---------------------')

    net_flatten, _ = flatten_layer(res5c)
    print('---------------------')
    print('Matrix dimension to the first FC layer')
    print(net_flatten.get_shape())
    print('---------------------')
    # net = fc_layer(net_flatten, h, 'FC1', batch_norm=True, add_reg=True, use_relu=True)
    # print('--------FC1-----------')
    # print(net.get_shape())
    # print('---------------------')
    # net = dropout(net, keep_prob)
    net = fc_layer(net_flatten, numClasses, 'FC1', batch_norm=True, add_reg=True, use_relu=False)
    print('--------FC2-----------')
    print(net.get_shape())
    print('---------------------')
    net = dropout(net, keep_prob)
    return net

def create_network34(X, h, keep_prob, numClasses):
    num_channels = X.get_shape().as_list()[-1]
    res1 = new_conv_layer1(inputs=X,
                           layer_name='res1',
                           stride=2,
                           num_inChannel=num_channels,
                           filter_size=7,
                           num_filters=64,
                           batch_norm=True,
                           use_relu=True)
    print('---------------------')
    print('X')
    print(X.get_shape())
    print('---------------------')
    print('---------------------')
    print('Before maxpool,Res1')
    print(res1.get_shape())
    print('---------------------')

    res1 = max_pool1(res1, ksize=3, stride=2, name='res1_max_pool')
    print('---------------------')
    print('After maxpool,Res1')
    print(res1.get_shape())
    print('---------------------')
    # Res2
    with tf.variable_scope('Res2'):
        res2a = bottleneck_block1(res1, 64, block_name='res2a',
                                 s1=1, k1=3, nf1=64, name1='res2a_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2a_branch2b',
                                 s3=1, k3=3, name3='res2a_branch1',first_block=False)

        print('Res2a')
        print(res2a.get_shape())
        print('---------------------')
        res2b = bottleneck_block1(res2a, 64, block_name='res2b',
                                 s1=1, k1=3, nf1=64, name1='res2b_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2b_branch2b',
                                 s3=1, k3=3, name3='res2b_branch1', first_block=False)
        print('Res2b')
        print(res2b.get_shape())
        print('---------------------')
        res2c = bottleneck_block1(res2b, 64, block_name='res2c',
                                  s1=1, k1=3, nf1=64, name1='res2c_branch2a',
                                  s2=1, k2=3, nf2=64, name2='res2c_branch2b',
                                  s3=1, k3=3, name3='res2c_branch1', first_block=False)
        print('Res2c')
        print(res2c.get_shape())
        print('---------------------')

    # Res3
    with tf.variable_scope('Res3'):
        res3a = bottleneck_block1(res2c, 64, block_name='res3a',
                                 s1=2, k1=3, nf1=128, name1='res3a_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3a_branch2b',
                                 s3=2, k3=3, name3='res3a_branch1', first_block=True)
        print('Res3a')
        print(res3a.get_shape())
        print('---------------------')
        res3b = bottleneck_block1(res3a, 128, block_name='res3b',
                                 s1=1, k1=3, nf1=128, name1='res3b_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3b_branch2b',
                                 s3=1, k3=3, name3='res3b_branch1', first_block=False)
        print('Res3b')
        print(res3b.get_shape())
        print('---------------------')
        res3c = bottleneck_block1(res3b, 128, block_name='res3c',
                                  s1=1, k1=3, nf1=128, name1='res3c_branch2a',
                                  s2=1, k2=3, nf2=128, name2='res3c_branch2b',
                                  s3=1, k3=3, name3='res3c_branch1', first_block=False)
        print('Res3c')
        print(res3c.get_shape())
        print('---------------------')
        res3d = bottleneck_block1(res3c, 128, block_name='res3d',
                                  s1=1, k1=3, nf1=128, name1='res3d_branch2a',
                                  s2=1, k2=3, nf2=128, name2='res3d_branch2b',
                                  s3=1, k3=3, name3='res3d_branch1', first_block=False)
        print('Res3d')
        print(res3d.get_shape())
        print('---------------------')


    # Res4
    with tf.variable_scope('Res4'):
        res4a = bottleneck_block1(res3d, 128, block_name='res4a',
                                 s1=2, k1=3, nf1=256, name1='res4a_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4a_branch2b',
                                 s3=2, k3=3, name3='res4a_branch1', first_block=True)
        print('---------------------')
        print('Res4a')
        print(res4a.get_shape())
        print('---------------------')
        res4b = bottleneck_block1(res4a, 256, block_name='res4b',
                                 s1=1, k1=3, nf1=256, name1='res4b_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4b_branch2b',
                                 s3=1, k3=3, name3='res4b_branch1', first_block=False)
        print('Res4b')
        print(res4b.get_shape())
        print('---------------------')
        res4c = bottleneck_block1(res4b, 256, block_name='res4c',
                                  s1=1, k1=3, nf1=256, name1='res4c_branch2a',
                                  s2=1, k2=3, nf2=256, name2='res4c_branch2b',
                                  s3=1, k3=3, name3='res4c_branch1', first_block=False)
        print('Res4c')
        print(res4c.get_shape())
        print('---------------------')
        res4d = bottleneck_block1(res4c, 256, block_name='res4d',
                                  s1=1, k1=3, nf1=256, name1='res4d_branch2a',
                                  s2=1, k2=3, nf2=256, name2='res4d_branch2b',
                                  s3=1, k3=3, name3='res4d_branch1', first_block=False)
        print('Res4d')
        print(res4d.get_shape())
        print('---------------------')
        res4e = bottleneck_block1(res4d, 256, block_name='res4e',
                                  s1=1, k1=3, nf1=256, name1='res4e_branch2a',
                                  s2=1, k2=3, nf2=256, name2='res4e_branch2b',
                                  s3=1, k3=3, name3='res4e_branch1', first_block=False)
        print('Res4e')
        print(res4e.get_shape())
        print('---------------------')
        res4f = bottleneck_block1(res4e, 256, block_name='res4f',
                                  s1=1, k1=3, nf1=256, name1='res4f_branch2a',
                                  s2=1, k2=3, nf2=256, name2='res4f_branch2b',
                                  s3=1, k3=3, name3='res4f_branch1', first_block=False)
        print('Res4f')
        print(res4f.get_shape())
        print('---------------------')

    # Res5
    with tf.variable_scope('Res5'):
        res5a = bottleneck_block1(res4f, 256, block_name='res5a',
                                 s1=2, k1=3, nf1=512, name1='res5a_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5a_branch2b',
                                 s3=2, k3=3, name3='res5a_branch1', first_block=True)
        print('---------------------')
        print('Res5a')
        print(res5a.get_shape())
        print('---------------------')
        res5b = bottleneck_block1(res5a, 512, block_name='res5b',
                                 s1=1, k1=3, nf1=512, name1='res5b_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5b_branch2b',
                                 s3=1, k3=3, name3='res5b_branch1', first_block=False)
        print('Res5b')
        print(res5b.get_shape())
        print('---------------------')
        res5c = bottleneck_block1(res5b, 512, block_name='res5c',
                                  s1=1, k1=3, nf1=512, name1='res5c_branch2a',
                                  s2=1, k2=3, nf2=512, name2='res5c_branch2b',
                                  s3=1, k3=3, name3='res5c_branch1', first_block=False)
        print('Res5c')
        print(res5c.get_shape())
        print('---------------------')

        res5c = avg_pool1(res5c, ksize=7, stride=1, name='res5_avg_pool')
        print('---------------------')
        print('Res5c after AVG_POOL')
        print(res5c.get_shape())
        print('---------------------')

    net_flatten, _ = flatten_layer(res5c)
    print('---------------------')
    print('Matrix dimension to the first FC layer')
    print(net_flatten.get_shape())
    print('---------------------')
    # net = fc_layer(net_flatten, h, 'FC1', batch_norm=True, add_reg=True, use_relu=True)
    # print('--------FC1-----------')
    # print(net.get_shape())
    # print('---------------------')
    # net = dropout(net, keep_prob)
    net = fc_layer(net_flatten, numClasses, 'FC1', batch_norm=True, add_reg=True, use_relu=False)
    print('--------FC1-----------')
    print(net.get_shape())
    print('---------------------')
    net = dropout(net, keep_prob)
    return net





def create_network101(X, h, keep_prob, numClasses):
    num_channels = X.get_shape().as_list()[-1]
    res1 = new_conv_layer1(inputs=X,
                           layer_name='res1',
                           stride=2,
                           num_inChannel=num_channels,
                           filter_size=7,
                           num_filters=64,
                           batch_norm=True,
                           use_relu=True)
    print('---------------------')
    print('X')
    print(X.get_shape())
    print('---------------------')
    print('---------------------')
    print('Before maxpool,Res1')
    print(res1.get_shape())
    print('---------------------')

    res1 = max_pool1(res1, ksize=3, stride=2, name='res1_max_pool')
    print('---------------------')
    print('After maxpool,Res1')
    print(res1.get_shape())
    print('---------------------')
    # Res2
    with tf.variable_scope('Res2'):
        res2a = bottleneck_block(res1, 64, block_name='res2a',
                                 s1=1, k1=1, nf1=64, name1='res2a_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2a_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res2a_branch2c',
                                 s4=1, k4=1, name4='res2a_branch1', first_block=True)
        print('Res2a')
        print(res2a.get_shape())
        print('---------------------')
        res2b = bottleneck_block(res2a, 256, block_name='res2b',
                                 s1=1, k1=1, nf1=64, name1='res2b_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2b_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res2b_branch2c',
                                 s4=1, k4=1, name4='res2b_branch1', first_block=False)
        print('Res2b')
        print(res2b.get_shape())
        print('---------------------')
        res2c = bottleneck_block(res2b, 256, block_name='res2c',
                                 s1=1, k1=1, nf1=64, name1='res2c_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res2c_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res2c_branch2c',
                                 s4=1, k4=1, name4='res2c_branch1', first_block=False)
        print('Res2c')
        print(res2c.get_shape())
        print('---------------------')

    # Res3
    with tf.variable_scope('Res3'):
        res3a = bottleneck_block(res2c, 256, block_name='res3a',
                                 s1=2, k1=1, nf1=128, name1='res3a_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3a_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res3a_branch2c',
                                 s4=2, k4=1, name4='res3a_branch1', first_block=True)
        print('Res3a')
        print(res3a.get_shape())
        print('---------------------')
        res3b = bottleneck_block(res3a, 512, block_name='res3b',
                                 s1=1, k1=1, nf1=128, name1='res3b_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3b_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res3b_branch2c',
                                 s4=1, k4=1, name4='res2b_branch1', first_block=False)
        print('Res3b')
        print(res3b.get_shape())
        print('---------------------')
        res3c = bottleneck_block(res3b, 512, block_name='res3c',
                                 s1=1, k1=1, nf1=128, name1='res3c_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3c_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res3c_branch2c',
                                 s4=1, k4=1, name4='res3c_branch1', first_block=False)
        print('Res3c')
        print(res3c.get_shape())
        print('---------------------')
        res3d = bottleneck_block(res3c, 512, block_name='res3d',
                                 s1=1, k1=1, nf1=128, name1='res3d_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res3d_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res3d_branch2c',
                                 s4=1, k4=1, name4='res3d_branch1', first_block=False)
        print('Res3d')
        print(res3d.get_shape())
        print('---------------------')

    # Res4
    with tf.variable_scope('Res4'):
        res4a = bottleneck_block(res3d, 512, block_name='res4a',
                                 s1=2, k1=1, nf1=256, name1='res4a_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4a_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4a_branch2c',
                                 s4=2, k4=1, name4='res4a_branch1', first_block=True)
        print('---------------------')
        print('Res4a')
        print(res4a.get_shape())
        print('---------------------')
        res4b = bottleneck_block(res4a, 1024, block_name='res4b',
                                 s1=1, k1=1, nf1=256, name1='res4b_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4b_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4b_branch2c',
                                 s4=1, k4=1, name4='res4b_branch1', first_block=False)
        print('Res4b')
        print(res4b.get_shape())
        print('---------------------')
        res4c = bottleneck_block(res4b, 1024, block_name='res4c',
                                 s1=1, k1=1, nf1=256, name1='res4c_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4c_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4c_branch2c',
                                 s4=1, k4=1, name4='res4c_branch1', first_block=False)
        print('Res4c')
        print(res4c.get_shape())
        print('---------------------')
        res4d = bottleneck_block(res4c, 1024, block_name='res4d',
                                 s1=1, k1=1, nf1=256, name1='res4d_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4d_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4d_branch2c',
                                 s4=1, k4=1, name4='res4d_branch1', first_block=False)
        print('Res4d')
        print(res4d.get_shape())
        print('---------------------')
        res4e = bottleneck_block(res4d, 1024, block_name='res4e',
                                 s1=1, k1=1, nf1=256, name1='res4e_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4e_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4e_branch2c',
                                 s4=1, k4=1, name4='res4e_branch1', first_block=False)
        print('Res4e')
        print(res4e.get_shape())
        print('---------------------')
        res4f = bottleneck_block(res4e, 1024, block_name='res4f',
                                 s1=1, k1=1, nf1=256, name1='res4f_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f_branch2c',
                                 s4=1, k4=1, name4='res4f_branch1', first_block=False)
        print('Res4f')
        print(res4f.get_shape())
        print('---------------------')

        res4f1 = bottleneck_block(res4f, 1024, block_name='res4f1',
                                 s1=1, k1=1, nf1=256, name1='res4f1_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f1_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f1_branch2c',
                                 s4=1, k4=1, name4='res4f1_branch1', first_block=False)
        print('Res4f1')
        print(res4f1.get_shape())
        print('---------------------')

        res4f2 = bottleneck_block(res4f1, 1024, block_name='res4f2',
                                 s1=1, k1=1, nf1=256, name1='res4f2_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f2_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f2_branch2c',
                                 s4=1, k4=1, name4='res4f2_branch1', first_block=False)
        print('Res4f2')
        print(res4f2.get_shape())
        print('---------------------')
        res4f3 = bottleneck_block(res4f2, 1024, block_name='res4f3',
                                 s1=1, k1=1, nf1=256, name1='res4f3_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f3_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f3_branch2c',
                                 s4=1, k4=1, name4='res4f3_branch1', first_block=False)
        print('Res4f3')
        print(res4f3.get_shape())
        print('---------------------')
        res4f4 = bottleneck_block(res4f3, 1024, block_name='res4f4',
                                 s1=1, k1=1, nf1=256, name1='res4f4_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f4_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f4_branch2c',
                                 s4=1, k4=1, name4='res4f4_branch1', first_block=False)
        print('Res4f4')
        print(res4f4.get_shape())
        print('---------------------')
        res4f5 = bottleneck_block(res4f4, 1024, block_name='res4f5',
                                 s1=1, k1=1, nf1=256, name1='res4f5_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f5_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f5_branch2c',
                                 s4=1, k4=1, name4='res4f5_branch1', first_block=False)
        print('Res4f5')
        print(res4f5.get_shape())
        print('---------------------')
        res4f6 = bottleneck_block(res4f5, 1024, block_name='res4f6',
                                 s1=1, k1=1, nf1=256, name1='res4f6_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f6_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f6_branch2c',
                                 s4=1, k4=1, name4='res4f6_branch1', first_block=False)
        print('Res4f6')
        print(res4f6.get_shape())
        print('---------------------')
        res4f7 = bottleneck_block(res4f6, 1024, block_name='res4f7',
                                 s1=1, k1=1, nf1=256, name1='res4f7_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f7_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f7_branch2c',
                                 s4=1, k4=1, name4='res4f7_branch1', first_block=False)
        print('Res4f7')
        print(res4f7.get_shape())
        print('---------------------')
        res4f8 = bottleneck_block(res4f7, 1024, block_name='res4f8',
                                 s1=1, k1=1, nf1=256, name1='res4f8_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f8_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f8_branch2c',
                                 s4=1, k4=1, name4='res4f8_branch1', first_block=False)
        print('Res4f8')
        print(res4f8.get_shape())
        print('---------------------')
        res4f9 = bottleneck_block(res4f8, 1024, block_name='res4f9',
                                 s1=1, k1=1, nf1=256, name1='res4f9_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f9_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f9_branch2c',
                                 s4=1, k4=1, name4='res4f9_branch1', first_block=False)
        print('Res4f9')
        print(res4f9.get_shape())
        print('---------------------')
        res4f10 = bottleneck_block(res4f9, 1024, block_name='res4f10',
                                 s1=1, k1=1, nf1=256, name1='res4f10_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f10_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f10_branch2c',
                                 s4=1, k4=1, name4='res4f10_branch1', first_block=False)
        print('Res4f10')
        print(res4f10.get_shape())
        print('---------------------')
        res4f11 = bottleneck_block(res4f10, 1024, block_name='res4f11',
                                 s1=1, k1=1, nf1=256, name1='res4f11_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f11_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f11_branch2c',
                                 s4=1, k4=1, name4='res4f11_branch1', first_block=False)
        print('Res4f11')
        print(res4f11.get_shape())
        print('---------------------')
        res4f12 = bottleneck_block(res4f11, 1024, block_name='res4f12',
                                 s1=1, k1=1, nf1=256, name1='res4f12_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f12_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f12_branch2c',
                                 s4=1, k4=1, name4='res4f12_branch1', first_block=False)
        print('Res4f12')
        print(res4f12.get_shape())
        print('---------------------')
        res4f13 = bottleneck_block(res4f12, 1024, block_name='res4f13',
                                 s1=1, k1=1, nf1=256, name1='res4f13_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f13_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f13_branch2c',
                                 s4=1, k4=1, name4='res4f13_branch1', first_block=False)
        print('Res4f13')
        print(res4f13.get_shape())
        print('---------------------')
        res4f14 = bottleneck_block(res4f13, 1024, block_name='res4f14',
                                 s1=1, k1=1, nf1=256, name1='res4f14_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f14_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f14_branch2c',
                                 s4=1, k4=1, name4='res4f14_branch1', first_block=False)
        print('Res4f14')
        print(res4f14.get_shape())
        print('---------------------')
        res4f15 = bottleneck_block(res4f14, 1024, block_name='res4f15',
                                 s1=1, k1=1, nf1=256, name1='res4f15_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f15_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f15_branch2c',
                                 s4=1, k4=1, name4='res4f15_branch1', first_block=False)
        print('Res4f15')
        print(res4f15.get_shape())
        print('---------------------')
        res4f16 = bottleneck_block(res4f15, 1024, block_name='res4f16',
                                 s1=1, k1=1, nf1=256, name1='res4f16_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f16_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f16_branch2c',
                                 s4=1, k4=1, name4='res4f16_branch1', first_block=False)
        print('Res4f16')
        print(res4f16.get_shape())
        print('---------------------')
        res4f17 = bottleneck_block(res4f16, 1024, block_name='res4f17',
                                 s1=1, k1=1, nf1=256, name1='res4f17_branch2a',
                                 s2=1, k2=3, nf2=256, name2='res4f17_branch2b',
                                 s3=1, k3=1, nf3=1024, name3='res4f17_branch2c',
                                 s4=1, k4=1, name4='res4f17_branch1', first_block=False)
        print('Res4f17')
        print(res4f17.get_shape())
        print('---------------------')

    # Res5
    with tf.variable_scope('Res5'):
        res5a = bottleneck_block(res4f17, 1024, block_name='res5a',
                                 s1=2, k1=1, nf1=512, name1='res5a_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5a_branch2b',
                                 s3=1, k3=1, nf3=2048, name3='res5a_branch2c',
                                 s4=2, k4=1, name4='res5a_branch1', first_block=True)
        print('---------------------')
        print('Res5a')
        print(res5a.get_shape())
        print('---------------------')
        res5b = bottleneck_block(res5a, 2048, block_name='res5b',
                                 s1=1, k1=1, nf1=512, name1='res5b_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5b_branch2b',
                                 s3=1, k3=1, nf3=2048, name3='res5b_branch2c',
                                 s4=1, k4=1, name4='res5b_branch1', first_block=False)
        print('Res5b')
        print(res5b.get_shape())
        print('---------------------')
        res5c = bottleneck_block(res5b, 2048, block_name='res5c',
                                 s1=1, k1=1, nf1=512, name1='res5c_branch2a',
                                 s2=1, k2=3, nf2=512, name2='res5c_branch2b',
                                 s3=1, k3=1, nf3=2048, name3='res5c_branch2c',
                                 s4=1, k4=1, name4='res5c_branch1', first_block=False)
        print('Res5c')
        print(res5c.get_shape())

        res5c = avg_pool1(res5c, ksize=7, stride=1, name='res5_avg_pool')
        print('---------------------')
        print('Res5c after AVG_POOL')
        print(res5c.get_shape())
        print('---------------------')

    net_flatten, _ = flatten_layer(res5c)
    print('---------------------')
    print('Matrix dimension to the first FC layer')
    print(net_flatten.get_shape())
    print('---------------------')
    # net = fc_layer(net_flatten, h, 'FC1', batch_norm=True, add_reg=True, use_relu=True)
    # print('--------FC1-----------')
    # print(net.get_shape())
    # print('---------------------')
    # net = dropout(net, keep_prob)
    net = fc_layer(net_flatten, numClasses, 'FC1', batch_norm=True, add_reg=True, use_relu=False)
    print('--------FC1-----------')
    print(net.get_shape())
    print('---------------------')
    net = dropout(net, keep_prob)
    return net
