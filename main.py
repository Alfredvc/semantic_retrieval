from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import uuid
import re
from PIL import Image
from glob import glob
from sklearn.decomposition import PCA
from collections import defaultdict

try:
    import cPickle as pickle
except:
    print('Using pickle')
    import pickle

import numpy as np
import tensorflow as tf

from inception_v3 import inception_v3
from inception_v3 import inception_v3_arg_scope
from tensorflow.python.training import saver as tf_saver
slim = tf.contrib.slim


class Network(object):
    """
    Convenience object to access a network.
    """

    def __init__(self, input_ph, output, end_points):
        self.input_ph = input_ph
        self.output = output
        self.end_points = end_points

    def eval(self, data, session, batch_size=32, output=None):
        if output is None:
            output = self.output
        result = []
        for chunk in chunks(data, batch_size):
            result.append(session.run(output, feed_dict={self.input_ph: chunk}))
        return np.vstack(result)


def create_network(num_outputs, inputs=None, scope='Network'):
    """
    Creates the final layers of the network, that maps the output from
    the Inception v3 network to the probability of each image.
    :param num_outputs:
    :param inputs:
    :param scope:
    :return:
    """
    if inputs is None:
        inputs = tf.placeholder('float32', [None, 2048])
    with tf.variable_scope(scope):
        end_points = {}
        end_point = 'FullyConnected_1'
        net = slim.fully_connected(inputs, 512, scope=end_point)
        end_points[end_point] = net

        end_point = 'Logits'
        net = slim.fully_connected(net, num_outputs, scope=end_point, activation_fn=None)
        end_points[end_point] = net

        end_point = 'Prediction'
        net = tf.nn.sigmoid(net, end_point)
        end_points[end_point] = net

        return Network(inputs, net, end_points)


def train_network(net, data, num_outputs, session, epochs=100, batch_size=32, validate=False, device='/cpu:0', run_name=None):
    if validate and run_name is None:
        run_name = 'second'+str(uuid.uuid4().get_hex()[:4])
    with tf.device(device):
        input_ph = net.input_ph
        output = net.output
        target_ph = tf.placeholder('float32', [None, num_outputs])
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(net.end_points['Logits'], target_ph))
        mse = tf.reduce_mean(tf.pow(target_ph-output, 2))*0.5
        train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)
    tf.scalar_summary('mean_squared_error', mse)
    tf.scalar_summary('cross_entropy', cross_entropy)
    merged = tf.merge_all_summaries()
    if validate:
        train_writer = tf.train.SummaryWriter('./summaries/run_{}_training'.format(run_name))

    #loss = tf.reduce_mean(tf.mul(target_ph, tf.log(net)))

    train_x = data['train']['inputs']
    train_y = data['train']['targets']
    if validate:
        validate_x = data['validate']['inputs']
        validate_y = data['validate']['targets']
        validate_writer = tf.train.SummaryWriter('./summaries/run_{}_validation'.format(run_name))

    session.run(tf.initialize_all_variables())
    print('Starting training')
    for epoch in range(0, epochs):
        mini_batches = 100 #int(train_x.shape[0]/batch_size)
        for i in range(0, mini_batches):
            indices = np.random.randint(0, train_x.shape[0]-1, batch_size)
            feed_dict = {input_ph:train_x[indices], target_ph:train_y[indices]}
            summary, _ = session.run([merged, train_step], feed_dict=feed_dict)
            if validate:
                train_writer.add_summary(summary, epoch*mini_batches + i)
        if validate:
            for val_x, val_y in zip(chunks(validate_x, batch_size), chunks(validate_y, batch_size)):
                feed_dict = {input_ph:val_x, target_ph:val_y}
                summary = session.run(merged, feed_dict=feed_dict)
                validate_writer.add_summary(summary, (epoch+1)*mini_batches)
        print('Finished epoch {}'.format(epoch))
    return net


def preprocess_image(image, size):
    # Store images as uint8 to reduce memory usage by 75%
    resized_img = np.asarray(image.resize((size, size), Image.ANTIALIAS), dtype=np.uint8)

    # Store images as float32 to maintain inception performance
    image = np.asarray(image.resize((299, 299), Image.ANTIALIAS), dtype=np.float32)

    return image, resized_img


def get_transformed_network_output(arr):
    """
    Transforms network output for an image to a boolean array representing
    its position in the tree.
    :param arr:
    :return:
    """
    arr = np.round(arr)
    arr = arr.clip(0, 1)
    arr = arr.astype('bool')
    return arr


def get_transformed_pca_output(arr):
    """
    Transforms pca-transformed data to a boolean array representing its position
    in the tree.
    :param arr:
    :return:
    """
    arr = np.copy(arr)
    arr += 0.5
    return get_transformed_network_output(arr)


def tf_transform_input_img(inputs):
    ph = tf.cast(inputs, dtype=tf.float32)
    ph = tf.mul(ph, (1.0/127.5))
    ph = tf.sub(ph, 1)
    return ph


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    :param l:
    :param n:
    :return: generator that yields chunks of size n from l
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_pca(data, n_components):
    """
    Trains a Principal Component Analysis model on the given data

    :param data:
    :param n_components: The number of components to use, should be larger than log2(len(data))
    :return: A sklearn.decomposition.PCA model trained on the given data
    """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca


def get_transformed_data(img_paths, size, checkpoint='./data/model.ckpt', num_classes=6012, batch_size=32,
                    name_regex=None):
    """
    Loads the supplied image_paths, runs them through the trained Inception_v3 network and returns
         a dictionary per image with it's bottleneck and a resized image.

    :param img_paths: iterable containing the paths to the images to be loaded
    :param size: image width and height in pixels
    :param checkpoint: location of the Inception_v3 weights
    :param num_classes:
    :param batch_size:
    :param name_regex: regex to obtain image name from image path
    :return:
    """
    if not os.path.exists(checkpoint):
        tf.logging.fatal(
            'Checkpoint %s does not exist. See README.md for more information',
            checkpoint)
    name_regex = name_regex if name_regex is not None else re.compile('.*\/.*\/(.*\.jpg)')
    g = tf.Graph()
    with g.as_default():
        input_images = tf.placeholder('float32', [None, 299, 299, 3])
        transformed_inputs = tf_transform_input_img(input_images)

        with slim.arg_scope(inception_v3_arg_scope()):
            logits, end_points = inception_v3(
                transformed_inputs, num_classes=num_classes, is_training=False)

        bottleneck = end_points['PreLogits']
        saver = tf_saver.Saver()
        data = {'image_names': [],
                'images': np.empty((len(img_paths), size, size, 3), dtype=np.uint8),
                'bottlenecks': np.empty((len(img_paths), 2048), dtype=np.float32)}
        i = 0
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            for c in chunks(img_paths, batch_size):
                batch = []
                inner_i = 0
                for file_name in c:
                    name = name_regex.match(file_name).group(1)
                    try:
                        image = Image.open(file_name)
                    except Exception:
                        print('Could not open image {}'.format(file_name))
                        continue
                    img, resized_img = preprocess_image(image, size)
                    batch.append(img)
                    data['images'][i+inner_i] = resized_img
                    data['image_names'].append(name)
                    inner_i += 1
                feed_dict = {input_images: np.asarray(batch)}
                # Run the evaluation on the image
                bottleneck_eval = sess.run(bottleneck, feed_dict=feed_dict)
                inner_i = 0
                for bn in bottleneck_eval:
                    # Resize the bottlenecks to the 0 - 1 range
                    data['bottlenecks'][i+inner_i] = (np.squeeze(bn) + 0.3) * (1.0/8.0)
                    inner_i += 1
                i += inner_i
            if i < len(img_paths):
                for key in ['images', 'bottlenecks']:
                    prev_shape = list(data[key].shape)
                    prev_shape[0] = i
                    data[key].resize(prev_shape)
            return data


def load_data(location, size, debug):
    """
    Loads the data needed to test and train the model, first checks files that cache the data
    if the files are not found, the data is generated.

    :param location: image folder path
    :param size: image size
    :param debug: Whether or not to add debugging information
    :return:
    """

    images_glob = '{}pics/*/*.jpg'.format(location)
    training_images = glob(images_glob)
    data_location = location.replace('train', 'data')
    try:
        print('Loading stored data...')
        data = {'train': {'image_names': np.load(data_location + 'train_img_names.npy')}}
        bn_location = data_location + 'train_bottlenecks.npy'
        if os.path.isfile(bn_location):
            data['train']['bottlenecks'] = np.load(bn_location)
        if debug:
            data['train']['images'] = np.load(data_location + 'train_imgs.npy')
            data['validate'] = {'images': np.load(data_location + 'validate_imgs.npy'),
                                'bottlenecks': np.load(data_location + 'validate_bottlenecks.npy'),
                                'image_names': np.load(data_location + 'validate_img_names.npy')}
    except Exception:
        print('Data not found, generating...')
        data = {'train': get_transformed_data(training_images, size)}

        # Don't create local variables to avoid polluting the function scope
        if debug:
            data['validate'] = get_transformed_data(glob('{}pics/*/*.jpg'.format(location.replace('train', 'validate'))),
                                                    size)
        np.save(data_location + 'train_imgs.npy', data['train']['images'])
        np.save(data_location + 'train_bottlenecks.npy', data['train']['bottlenecks'])
        np.save(data_location + 'train_img_names.npy', data['train']['image_names'])

        np.save(data_location + 'validate_imgs.npy', data['validate']['images'])
        np.save(data_location + 'validate_bottlenecks.npy', data['validate']['bottlenecks'])
        np.save(data_location + 'validate_img_names.npy', data['validate']['image_names'])
    return data


def load_pca(location, data, data_count, n_components):
    """
    Creates the tree structure for the given data
    :param location:
    :param data:
    :param data_count:
    :param n_components:
    :return:
    """
    try:
        print('Loading stored PCA model...')
        pca = None
        with open(location+'/pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
        transformed_data = np.load(location+'/pca_data.npy')
        return pca, transformed_data
    except Exception:
        print('PCA model not found, generating...')
        pca_data = data['train']['bottlenecks']
        pca = get_pca(pca_data[:data_count], n_components)
        transformed_data = pca.transform(pca_data)
        transformed_data = get_transformed_pca_output(transformed_data)
        with open(location+'/pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)
        np.save(location+'/pca_data.npy', transformed_data)
        return pca, transformed_data


def create_lookup(data, tags):
    """
    Creates a dictionary, and a mapping function such that mapping an image's location in the tree
    with the function and using it as a key to the dictionary will return the image id.
    :param data:
    :param tags:
    :return:
    """
    powers_of_two = np.asarray([2**i for i in reversed(range(data.shape[1]))])

    def binary_to_idx(binary):
        return np.sum(powers_of_two[binary])

    lookup = defaultdict(list)
    for d, t in zip(data, tags):
        idx = binary_to_idx(d)
        if t not in lookup[idx]:
            lookup[idx].append(t)
    return lookup, binary_to_idx


def get_top_n(probs, data, tags, n):
    """
    Gets the n images with highest probability
    :param probs: probabilities
    :param data: the image bottlenecks
    :param tags: the identification of the images
    :param n: the amount of images to return
    :return:
    """
    probs_ = data * probs + np.logical_not(data) * (1 - probs)
    return tags[np.argsort(np.prod(probs_, axis=1))[-n:]]


def create_combined_network(n_components):
    """
    Combines the Inception v3 network with my own
    :param n_components:
    :return:
    """
    print('Creating inception_v3 network')
    input_images = tf.placeholder('float32', [None, 299, 299, 3])
    transformed_inputs = tf_transform_input_img(input_images)
    with slim.arg_scope(inception_v3_arg_scope()):
        logits, end_points = inception_v3(
            transformed_inputs, num_classes=6012, is_training=False)
    bottleneck = slim.flatten((end_points['PreLogits'] + 0.3) * (1.0/8.0))
    print('Creating lower network')
    network = create_network(n_components, bottleneck)

    return Network(input_images, network.output, network.end_points)


def train(location='./train/'):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that is required for testing the model
    must be saved to file (e.g., pickle) so that the test procedure can load, execute and report
    :param location: The location of the training data folder hierarchy
    :return: nothing
    """

    inception_checkpoint = './data/model.ckpt'
    lower_checkpoint = './data/lower_graph.checkpoint'
    combined_model_checkpoint = './data/model.checkpoint'
    size = 128
    n_components = 40
    device = '/cpu:0'
    debug = False

    data = load_data(location, size, debug)
    pca, transformed_data = load_pca('./data'.format(n_components), data, 1000, n_components)
    second_data = {'train':
                       {'inputs': data['train']['bottlenecks'],
                        'targets': transformed_data}
                   }

    if debug:
        second_data['validate'] = {
                           'inputs': data['validate']['bottlenecks'],
                           'targets': get_transformed_pca_output(pca.transform(data['validate']['bottlenecks']))}

    with tf.device(device):
        g = tf.Graph()
        with g.as_default():
            with tf.Session() as sess:
                print('Training lower network')
                network = create_network(n_components)
                network = train_network(network, second_data, n_components,sess, epochs=100, validate=debug, device=device)
                saver = tf_saver.Saver()
                print('Saving lower network')
                saver.save(sess, lower_checkpoint)

        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()

            network = create_combined_network(n_components)

            inception_scope = 'InceptionV3'
            lower_scope = 'Network'

            all_vars = tf.all_variables()
            inception_vars = [k for k in all_vars if k.name.startswith(inception_scope)]
            lower_vars = [k for k in all_vars if k.name.startswith(lower_scope)]

            print('Loading inception_v3 network')
            tf_saver.Saver(inception_vars).restore(sess, inception_checkpoint)
            print('Loading lower network')
            tf_saver.Saver(lower_vars).restore(sess, './data/lower_graph.checkpoint')
            print('Saving combined network')
            tf_saver.Saver().save(sess, combined_model_checkpoint)


def test(queries=list(), location='./test'):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned
    :param queries: list of image-IDs. Each element is assumed to be an entry in the test set. Hence, the image
    with id <id> is located on my computer at './test/pics/<id>.jpg'. Make sure this is the file you work with...
    :param location: The location of the test data folder hierarchy
    :return: a dictionary with keys equal to the images in the queries - list, and values a list of image-IDs
    retrieved for that input
    """

    model_checkpoint = './data/model.checkpoint'
    size = 128
    n_components = 40
    device = '/cpu:0'
    debug = False

    data = load_data('./train/', size, debug)

    pca, transformed_data = load_pca('./data'.format(n_components), data, 1000, n_components)

    my_return_dict = {}

    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        with tf.device(device):
            network = create_combined_network(n_components)
            tf_saver.Saver().restore(sess, model_checkpoint)

            query_images = []
            print('Loading images...')
            for query in queries:
                file_name = '{}/pics/{}.jpg'.format(location, query)
                try:
                    image = Image.open(file_name)
                    query_images.append(np.asarray(image.resize((299, 299), Image.ANTIALIAS), dtype=np.float32))
                except Exception:
                    print('Could not open image {}'.format(file_name))
                    continue

            print('Running images through model...')
            net_outputs = network.eval(query_images, sess)

            print('Generating candidates')
            for i, query in enumerate(queries):
                net_output = net_outputs[i]
                candidates = get_top_n(net_output, transformed_data, data['train']['image_names'], 200)
                try:
                    candidate_names = [c.replace('.jpg','') for c in candidates]
                except:
                    candidates = [c.decode('UTF-8') for c in candidates]
                    candidate_names = [c.replace('.jpg','') for c in candidates]
                my_return_dict[query] = candidate_names

            return my_return_dict