import numpy as np
import tensorflow as tf
import scipy
import datetime as dt

version = '1.0'

def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 

def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

class Convnet:
    def __init__(self, restore = True, save = True,image_size = [400,400], savePath = 'model/',resize = True, form = False):
        f1, f2, f3 = 32, 64, 128
        self.image_size = image_size
        self.iters = 0
        self.images_seen = 0
        self.images_right = 0
        self.save = save
        self.resize = resize
        self.savePath = savePath
        self.allSeen = []
        if form:
            with tf.variable_scope('format') as scope:
                self.shape_placeholder = tf.placeholder(tf.int32, shape = None)
                self.form_placeholder = tf.placeholder(tf.int32, shape = [None,None,3])
                
                img = tf.image.resize_images(self.form_placeholder,self.shape_placeholder)
                self.formatedImages = tf.image.resize_image_with_crop_or_pad(img, self.image_size[1], self.image_size[0])


        with tf.variable_scope('prepare') as scope:
            self.prep_placeholder = tf.placeholder(tf.float32, shape = [None, 3, None, None])
        images = tf.transpose(self.prep_placeholder,[0,2,3,1])
        resized_images = tf.image.resize_images(images, self.image_size)
        self.resized_images = tf.cast(resized_images, np.uint8)
        with tf.variable_scope('input'):
            self.image_placeholder = tf.placeholder(tf.float32, shape = [None, image_size[0], image_size[1], 3], name='image')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
            self.is_valid = tf.placeholder(tf.float32,shape = [None,1])
        with tf.variable_scope('conv') as scope:           
            conv1 = tf.layers.conv2d(self.image_placeholder,f1,kernel_size = [5,5],strides=[2,2],padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='conv1')
            bn1 = tf.contrib.layers.batch_norm(conv1, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
            act1 = lrelu(conv1, n='act1')
            conv2 = tf.layers.conv2d(act1,f2,kernel_size = [5,5],strides=[2,2],padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn2')
            act2 = lrelu(conv2, n='act2')
            conv3 = tf.layers.conv2d(act2,f3,kernel_size = [5,5],strides=[2,2],padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn3')
            act3 = lrelu(conv3, n='act3')
            dim = int(np.prod(act3.get_shape()[1:]))
            fc1 = tf.reshape(act3, shape=[-1, dim], name = 'fcl')
            w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        self.result = logits
        self.loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.result,self.is_valid))
        t_vars = tf.trainable_variables()
        Vars = [var for var in t_vars if 'conv' in var.name]
        self.trainer = tf.train.AdamOptimizer(1e-4).minimize(self.loss, var_list=Vars)
        self.clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in Vars]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if save or restore:
            self.saver = tf.train.Saver()
        if restore:
            try:
                self.saver.restore(self.sess,tf.train.latest_checkpoint(savePath))
            except: print("Can't load model at "+savePath)
        tf.Graph.finalize(self.sess)
    
    def to_image_mat(self, images, form):
        if form == 'path':
            images = scipy.ndimage.imread(images)
            form = 'imageMat'
        if form == 'paths':
            images = [scipy.ndimage.imread(image) for image in images]
        return images
        
    def format_images(self, images,form = 'imageMat'):
        if form != 'imageMat':
            images = self.to_image_mat(images, form)
        formatedImages = []
        for img in images:
            shape = img.shape[0:2]
            newWidth = (self.image_size[0] / shape[0]) * shape[1] 
            newShape = [self.image_size[0],newWidth]
            newImage = self.sess.run([self.formatedImages], feed_dict={self.form_placeholder:img, self.shape_placeholder:newShape})[0]
            formatedImages.append(newImage)
        return formatedImages

    def prepare_image(self, images, form = 'imageMat'):
        if form in ['path','paths']:
            images = self.to_image_mat(images, form)
            form = 'imageMat'
        if form == 'imageMat':
            if 1:
                images = np.array(images)
                if len(images.shape) == 3:
                    images = np.array([images])
                newDic = np.zeros(shape=(len(images),3,images.shape[1],images.shape[2]))
                for x in range(len(images)):
                    newDic[x-1] = images[x].reshape(3,images.shape[1],images.shape[2])
                resized_images = self.sess.run([self.resized_images], feed_dict = {self.prep_placeholder: newDic})
                resized_images = resized_images[0]
            else:
                print(e)
                print(images.shape)
                return 'Invalid Shape'
        else:
            resized_images = images
        
        return resized_images
        
    def forward_only(self,image, form = 'imageMat'):
        if form != 'resized': resized_image = self.prepare_image(image, form)
        else: resized_image = image
        result = self.sess.run([self.result], feed_dict = {self.image_placeholder: resized_image})
        return result

    def train(self, images,valid, form = 'imageMat', saveIters = 100):
        if form != 'resized': resized_image = self.prepare_image(images, form)
        else: resized_image = images
        self.sess.run(self.clip)
        _, loss,result = self.sess.run([self.trainer, self.loss, self.result], feed_dict = {self.image_placeholder: resized_image, self.is_valid: valid})
        for r in range(len(result)):
            if (result[r] > 0 and valid[r] == 1) or (result[r] < 0 and valid[r] == 0):
                self.allSeen.append(1)
                self.images_right += 1
            else:
                self.allSeen.append(0)
            self.images_seen += 1
        if self.iters > 0 and self.iters % saveIters == 0 and self.save:
            print('Saving model...')
            self.saver.save(self.sess, self.savePath+ str(dt.datetime.now()).split('.')[0].replace(':','.'), write_meta_graph=False)  
        self.iters += 1
        return loss
    def current_efficiency(self):
        return str(sum(self.allSeen[-100:]))+'%'
    def efficiency(self):
        return str(int((self.images_right/self.images_seen)*100)) + '%'
        
            
