import tensorflow as tf

def ResidualBlock(inputs, inputs0, num_filters, nfs, name):
    outputs = tf.layers.conv2d(inputs, num_filters, nfs, padding='SAME',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name=name + '_conv1', use_bias=False)
    outputs = tf.nn.relu(outputs)

    outputs = tf.layers.conv2d(outputs, num_filters, nfs, padding='SAME',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name=name + '_conv2', use_bias=False)
    outputs = tf.nn.relu(outputs)


    outputs = tf.layers.conv2d(outputs, num_filters, nfs, padding='SAME',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name=name + '_conv3', use_bias=False)
    outputs = tf.nn.relu(outputs+inputs0)
    return outputs

def Res_Model(inputs, inputs0):

    outputs = tf.layers.conv3d( inputs, 1, (4,1,1), padding="valid", 
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                name='conv1', use_bias=False)
    outputs = tf.squeeze(outputs, 1)

    outputs = ResidualBlock(outputs,inputs0, 64, 7, 'resblock1')
    outputs = ResidualBlock(outputs,outputs, 64, 7, 'resblock2')
    outputs = ResidualBlock(outputs,outputs, 64, 5, 'resblock3')
    outputs = ResidualBlock(outputs,outputs, 64, 5, 'resblock4')
    outputs = ResidualBlock(outputs,outputs, 64, 3, 'resblock5')
    outputs = ResidualBlock(outputs,outputs, 64, 3, 'resblock6')
    outputs = ResidualBlock(outputs,outputs, 64, 3, 'resblock7')
    outputs = ResidualBlock(outputs,outputs, 64, 3, 'resblock8')  
    outputs = ResidualBlock(outputs,outputs, 64, 3, 'resblock9')   

    outputs = tf.layers.conv2d(outputs, 32, 3, padding='SAME', 
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                               name='conv3', use_bias=False)
    outputs = tf.nn.relu(outputs)

    
    outputs = tf.layers.conv2d(outputs,  1, 3, padding='SAME', 
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                               name='conv4', use_bias=False)
    outputs = tf.nn.relu(outputs)

    return outputs