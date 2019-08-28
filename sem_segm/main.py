import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.2'), 'Please use TensorFlow version 1.2 or newer.  You are using {}'.format(tf.__version__) 
print('TensorFlow Version: {}'.format(tf.__version__)) 

# Check for a GPU, here use the google colab for its free 
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    """
   
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input= tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name) 
    keep_prob= tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name) 
    layer3_out= tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name) 
    layer4_out=tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name) 
    layer7_out= tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name) 
    
    return image_input, keep_Prob, layer3_out, layer4_out, layer7_out 

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    vgg_layer3_out: TF Tensor for VGG Layer 3 output
    vgg_layer4_out: TF Tensor for VGG Layer 4 output
    vgg_layer7_out: TF Tensor for VGG Layer 7 output
    num_classes: Number of classes to classify 
    """
    layer7a_out= tf.layers.conv2d(vgg_layer7_out, num_class, 1, 
                                  padding='same',
                                  kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3)) 
    
    #unsample, transposed convolutional layer 
    layer4a_in1= tf.layers.conv2d_transpose(layer7a_out, num_classes, 4, 
                                            strides=(2,2),
                                            padding= 'same',
                                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3)) 
    layer4a_in2= tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                 padding='same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3)) 
    
    #use skip connections 
    layer4a_out= tf.add(layer4a_in1, layer4a_in2) 
    
    #unsample
    layer3a_in1= tf.layers.conv2d_transpose(layer4a_out, num_classes, 4,
                                           strides=(2,2),
                                           padding='same',
                                           kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3)) 
    
    #1x1 conv layers for vgg layer3 
    layer3a_in2= tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                 padding='same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3)) 
    
    
    #use skip connections 
     layer3a_out= tf.add(layer3a_in1, layer3a,in2)
    
    #unsample 
    nn_last_layer= tf.layers.conv2d_transpose(layer3a_out, num_classes, 16,
                                             strides= (8,8),
                                             padding='same',
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer= tf.random.layers.l2_regularizer(1e-3)) 
    return nn_last_layers(layers)                            
                                 
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    nn_last_layer: TF Tensor of the last layer in the neural network
    correct_label: TF Placeholder for the correct label image
    learning_rate: TF Placeholder for the learning rate
    num_classes: Number of classes to classify 
    """ 
    #make the logits a 2d tensor where each row represents a pixel and each column a class 
    logits= tf.reshape(nn_last_layer, (-1, num_classes)) 
    correct_label= tf.reshape(correct_label, (-1, num_claeees)) 
    
    #define the loss fn
    cross_entropy_loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    #define training ops 
    optimizer= tf.train.AdamOptimizer(learning_rate= learning_rate) 
    train_op= optimizer.minimize(cross_entropy_loss) 
    
    return logits, train_op, cross_entropy_loss 

tests.test_optimize(optimize) 


def train_nn(sess, epochs, bs, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate): 
    """
    Train neural network and print out the loss during training.
    the fn here is become the param to another fn, it proofs again everything in python is objetc, 
    get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(bs)
    train_op: TF Operation to train the neural network
    cross_entropy_loss: TF Tensor for the amount of loss
    input_image: TF Placeholder for input images 
    """
    sess.run(tf.global_variables_initializer()) 
    
    print('starting training')
    
    for i in range(epochs):
        print('EPOCH {}'.format(i+1))
        for image, label in get_batches_fn(bs):
            _,loss, = sess.run([train_op, cross_entropy_loss],
                                feed_dict= {input_image:image, corect_label: label,
                                           0.5, learning_rate:0.0009})
            print('loss:{:.3f}'.format(loss)) 
            
        
tests.test_train_nn(train_nn) 

def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir) 

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape) 
        #sometimes the better result can be generated by image augumentation 
        epochs= 50
        bs=5
        
        correct_label= tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label') 
        learning_rate= tf.placeholder(tf.float32, name='lr')
        input_image, keep_prob, vgg_layers_out, b=vgg_layer4_out, vgg_layer7_out= load_vgg(sess,vgg_path) 
        nn_last_layer= layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes) 
        logits, train_op, cross_entropy_loss= optimize(nn_last_layer, correct_label, learning_rate, num_classes) 

        # Train NN using the train_nn function
         train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
         helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image) 


if __name__ == '__main__':
    run() 