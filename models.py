import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import cv2
import os
from imutils import paths
import imutils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class SimplePreprocessor:
    def __init__(self, width, height, inter= cv2.INTER_AREA):
        self.width= width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        image= cv2.resize(image, (self.width, self.height), interpolation = self.inter)
        b,g,r= cv2.split(image)
        image= cv2.merge((r,g,b))
        return image
    
    
class AnimalsDatasetManager:
    def __init__(self, preprocessors=None, random_state=6789):
        self.random = np.random.RandomState(random_state)
        self.preprocessors = preprocessors
        # self.preprocessors is a list of preprocessor for data augmentation
        # it can be an instance of SimplePreprocessor, which performs resizing image and re-orders the channels to RGB
        if self.preprocessors is None:
            self.preprocessors = list()
    
    def load(self, label_folder_dict, max_num_images=500, verbose =-1):
        # label_folder_dict: a dict mapping label to folder path
        data =list(); labels = list()
        for label, folder in label_folder_dict.items():
            image_paths = list(paths.list_images(folder)) # get the list of paths to all images in the folder
            print(label, len(image_paths))
            for (i, image_path) in enumerate(image_paths):
                image = cv2.imread(image_path)
                #if preprocessing images
                if self.preprocessors is not None: 
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                data.append(image); labels.append(label)
                if verbose > 0 and i>0 and (i+1)% verbose ==0:
                    print("Processed {}/{}".format(i+1, max_num_images))
                if i+1 >= max_num_images:
                    break
        self.data= np.array(data)
        self.labels= np.array(labels)
        self.train_size= int(self.data.shape[0])
    
    def process_data_label(self):
        label_encoder= preprocessing.LabelEncoder()
        label_encoder.fit(self.labels)
        self.labels= label_encoder.transform(self.labels)
        self.data= self.data.astype("float") / 127.5 - 1 # standardize pixel value to range [-1, 1]
        self.classes= label_encoder.classes_
    
    def train_valid_test_split(self, train_size=0.8, test_size= 0.1, rand_seed=33):
        valid_size = 1 - (train_size + test_size)
        X1, X_test, y1, y_test = train_test_split(self.data, self.labels, test_size = test_size, random_state= rand_seed)
        self.X_test= X_test
        self.y_test= y_test
        X_train, X_valid, y_train, y_valid = train_test_split(X1, y1, test_size = float(valid_size)/(valid_size+ train_size))
        self.X_train= X_train
        self.y_train= y_train
        self.X_valid= X_valid
        self.y_valid= y_valid
    
    def next_batch(self, batch_size=32):
        idx = self.random.choice(self.X_train.shape[0], batch_size, replace=batch_size > self.X_train.shape[0])
        return self.X_train[idx], self.y_train[idx]
        
he_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN')
normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
zero_initializer = tf.constant_initializer(0.0)

class Layers:
    @staticmethod
    def dense(inputs, output_dim, act=None, name='dense'):
        """
        Dense layer
    
        inputs: of shape [batch size, D]
        output_dim: The number of hidden units in the output layer
        act: Apply activation function if act is not None
        
        Return: a tensor of shape [batch size, output_dim]
        """
        with tf.variable_scope(name):
            W = tf.get_variable('W', [inputs.get_shape()[1], output_dim], initializer=he_initializer)
            b = tf.get_variable('b', [output_dim], initializer=zero_initializer)
            Wxb= tf.matmul(inputs, W) + b
            return Wxb if act is None else act(Wxb)

    @staticmethod
    def conv2D(inputs, output_dim, kernel_size=3, strides=1, padding="SAME", act=None, name= "conv"):
        """
        Convolutional layer
        
        inputs: a feature map of shape [batch size, H, W, C]
        output_dim: the number of feature maps in the output
        kernel_size: a tuple (h, w) specifying the heigh and width of the convolution window.
                     Can be a integer if the heigh and width are equal
        strides: a tuple (h, w) specifying the strides of the convolution along the height and width.
                 Can be a integer if the stride is the same along the height and width
        act: Apply activation function if act is not None
        
        Return: a tensor of shape [batch_size, H', W', C']
        """
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        strides = (strides, strides) if isinstance(strides, int) else strides
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                [kernel_size[0], kernel_size[1], inputs.get_shape()[-1], output_dim],
                                initializer=he_initializer)
            b = tf.get_variable('b', [output_dim], initializer=zero_initializer)
            conv = tf.nn.conv2d(input=inputs, filter=W, strides=[1, strides[0], strides[1], 1], padding= padding)
            conv = conv + b
            return conv if act is None else act(conv)
    @staticmethod
    def max_pool(inputs, ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name='max_pool'):
        return tf.nn.max_pool(value= inputs, ksize=ksize, strides= strides, padding= padding, name=name)
    
    @staticmethod
    def mean_pool(inputs, ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME", name='avg_pool'):
        return tf.nn.avg_pool(value= inputs, ksize=ksize, strides= strides, padding= padding, name=name)
    
    @staticmethod
    def dropout(inputs, keep_prob, name='drop_out'):
        return tf.nn.dropout(inputs, keep_prob= keep_prob, name=name)
    
    @staticmethod
    def batch_norm(inputs, phase_train, name='batch_norm'):
        return tf.layers.batch_norm(inputs, momentum=0.9, epsilon=1e-5, training=phase_train, name=name)
    
class DefaultModel():
    def __init__(self,
                 name='network1',
                 width=32, height=32, depth=3,
                 num_blocks=2,
                 feature_maps=32,
                 num_classes=4, 
                 keep_prob=1.0,
                 batch_norm=None,
                 activation_func=tf.nn.relu,
                 optimizer='adam',
                 batch_size=10,
                 num_epochs= 20,
                 learning_rate=0.0001,
                 random_state=6789):
        assert (1 << num_blocks <= min(width, height))
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        self.num_blocks = num_blocks
        self.feature_maps = [feature_maps * (1 << i) for i in range(num_blocks)]
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.batch_norm = batch_norm
        self.activation_func = activation_func
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_state = random_state
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        
        self.tf_graph = tf.Graph()
        self.session = tf.Session(graph=self.tf_graph)
        with self.tf_graph.as_default():
            if random_state is not None:
                tf.set_random_seed(random_state)
                np.random.seed(random_state)
            
            self.build()
            self.tf_merged_summaries = tf.summary.merge_all()
        
        # create log_path
        self.root_dir = 'models/{}'.format(self.name)
        self.log_path = os.path.join(self.root_dir, 'logs')
        self.model_path = os.path.join(self.root_dir, 'saved/model.ckpt')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)            
            
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))

        self.tf_summary_writer = tf.summary.FileWriter(self.log_path, self.session.graph)
        
    def build(self):
        self.X = tf.placeholder(shape=[None, self.height, self.width, self.depth], dtype=tf.float32)
        tf.summary.image('input_image_sample', self.X, max_outputs=5) # visualize some input images
        self.y = tf.placeholder(shape=[None], dtype=tf.int64) # label
        self.keep_prob_holder = tf.placeholder(dtype=tf.float32) # to be passed to dropout layer
        self.phase_train = tf.placeholder(dtype= tf.bool) # to be pass to batch norm layer
        
        logits = self.build_cnn(self.X, name='CNN')

        with tf.name_scope("train"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
            self.loss = tf.reduce_mean(cross_entropy)
        
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): # need to run update ops when using tf.layers.batch_normalization
                grads = self.optimizer.compute_gradients(self.loss, var_list=params)
                self.train = self.optimizer.apply_gradients(grads)

            # visualize weight values, gradients, and gradient norms
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/values', var)
                    tf.summary.histogram(var.op.name + '/gradients', grad)
                    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(grad)))
                    tf.summary.scalar(var.op.name + '/gradient_norm', gradient_norm)
        
        with tf.name_scope("predict"):
            self.y_pred = tf.argmax(logits, 1)
            corrections = tf.equal(self.y_pred, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
            
        self.session.run(tf.global_variables_initializer())
        
    def build_cnn(self, x, reuse=False, name='CNN'):
        with tf.variable_scope(name, reuse=reuse):
            # first block
            h = Layers.conv2D(x, 32, kernel_size=3, strides=1, name='block0.conv0')
            tf.summary.histogram('block0.conv0', h) # visualize pre-activation value
            h = self.activation_func(h, name='block0.relu0')
            tf.summary.histogram('block0.relu0', h) # visualize post-activation value
            
            h = Layers.conv2D(h, 32, kernel_size=3, strides=1, name='block0.conv1')
            tf.summary.histogram('block0.conv1', h)
            h = self.activation_func(h, name='block0.relu1')
            tf.summary.histogram('block0.relu1', h)
            
            h = Layers.mean_pool(h, name='block0.avg_pool')
            
            # second block
            h = Layers.conv2D(h, 64, kernel_size=3, strides=1, name='block1.conv0')
            tf.summary.histogram('block1.conv0', h)
            h = self.activation_func(h, name='block1.relu0')
            tf.summary.histogram('block1.relu0', h)
            
            h = Layers.conv2D(h, 64, kernel_size=3, strides=1, name='block1.conv1')
            tf.summary.histogram('block1.conv1', h)
            h = self.activation_func(h, name='block1.relu1')
            tf.summary.histogram('block1.relu1', h)
            h = Layers.mean_pool(h, name='block0.avg_pool')            
            
            # after two block
            # reshape and use a fully connected layer
            h = tf.reshape(h, [-1, h.get_shape()[1] * h.get_shape()[2] * h.get_shape()[3]])
            
            # now output the logits for softmax classification
            logits = Layers.dense(h, self.num_classes, name='output.logits')
            tf.summary.histogram('output.logits', logits)
            return logits
    
    def partial_fit(self, X_batch, y_batch, summary=False, epoch=0):
        feed_dict={self.X:X_batch, self.y:y_batch, self.keep_prob_holder: self.keep_prob, self.phase_train: True}
        if summary:
            _, _summary = self.session.run([self.train, self.tf_merged_summaries], feed_dict=feed_dict)
            self.tf_summary_writer.add_summary(_summary, epoch)
        else:
            self.session.run([self.train], feed_dict=feed_dict)
    
    def fit(self, data_manager, batch_size=None, num_epochs=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        num_epochs = self.num_epochs if num_epochs is None else num_epochs
        iter_per_epoch= data_manager.X_train.shape[0] // batch_size + 1
        
        # initialize list to store history of training
        self.best_val_accuracy = 0
        self.train_losses, self.train_accs, self.val_losses, self.val_accs = [], [], [], []
        
        for epoch in range(num_epochs):
            for i in range(iter_per_epoch):
                X_batch,y_batch= data_manager.next_batch(batch_size)
                # run summary to visualize in Tensorboard only at the end of each epoch to save time and memory
                self.partial_fit(X_batch, y_batch, summary=i == iter_per_epoch - 1, epoch=epoch)
                
            train_loss, train_acc= self.compute_acc_loss(data_manager.X_train, data_manager.y_train)
            val_loss, val_acc= self.compute_acc_loss(data_manager.X_valid, data_manager.y_valid)
            
            # save the best model using validation accuracy as criterion
            if val_acc > self.best_val_accuracy:
                self.save(self.model_path)
                self.best_val_accuracy = val_acc
            
            # store progress
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            print("Epoch {:03d}: train loss={:.4f}, val loss={:.4f}\n".format(epoch, train_loss, val_loss))
            print("         : train acc={:.4f}, val acc={:.4f}\n".format(train_acc, val_acc))
            
        print("Finish training and come to testing")
        # load the best model
        self.load(self.model_path)
        # and then test
        y_test, acc= self.predict(data_manager.X_test, data_manager.y_test)
        print("Testing accuracy= {:.4f}".format(acc))
        
    def save(self, model_path):
        with self.tf_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, model_path)       
            
    def load(self, model_path):
        with self.tf_graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, model_path)       
        
    def predict(self, X, y):
        y_pred, acc= self.session.run([self.y_pred, self.accuracy], feed_dict={self.X:X, self.y:y, self.keep_prob_holder:1,
                                                                              self.phase_train: False})
        return y_pred, acc
        
    def compute_acc_loss(self, X, y):
        loss, acc = self.session.run([self.loss, self.accuracy], 
                                             feed_dict={self.X:X, self.y:y, self.keep_prob_holder:1, self.phase_train: False})
        return loss, acc
        
    def plot_progress(self):
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        ax[0].plot(np.arange(len(self.train_accs)), self.train_accs, "g")
        ax[0].plot(np.arange(len(self.val_accs)), self.val_accs, "b")
        ax[0].set_title('Accuracy over epoch')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend(['Train Accuracy', 'Val Accuracy'], loc='best')

        ax[1].plot(np.arange(len(self.train_losses)), self.train_losses, "g")
        ax[1].plot(np.arange(len(self.val_losses)), self.val_losses, "b")
        ax[1].set_title('Loss over epochs')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend(['Train Loss', 'Val Loss'], loc='best')
        plt.show()
        
    def plot_prediction(self, x, y, classes, tile_shape=(5, 5)):
        y_pred, acc = self.predict(x, y)
        plt.clf()
        fig, ax = plt.subplots(tile_shape[0], tile_shape[1], figsize=(2 * tile_shape[1], 2 * tile_shape[0]))
        idx = np.random.choice(len(y_pred), tile_shape[0] * tile_shape[1])

        for i in range(tile_shape[0]):
            for j in range(tile_shape[1]):
                ax[i, j].imshow((x[idx[i * tile_shape[1] + j]] + 1.0)/2)
                ax[i, j].set_title('{} (p: {})'.format(classes[y[idx[i * tile_shape[1] + j]]],
                                                        classes[y_pred[idx[i * tile_shape[1] + j]]]))
                ax[i, j].grid(False)
                ax[i, j].axis('off')
        plt.show()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
