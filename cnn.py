import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from mnist import MNIST


def binarization(pixels):
	thres = 100

	coord = []
	dim = pixels.shape

	flat_pixels = pixels.flatten()

	for i in range(len(flat_pixels)):
		if flat_pixels[i] == 0:
			flat_pixels[i] = -1

	pixels = np.array(flat_pixels, dtype='float').reshape(dim)

	for i in range(dim[0]):
		for j in range(dim[1]):
			if pixels[i, j] > thres:
				coord.append([i, j])

	return coord

def prep_data(img):
	img = np.array(img, dtype='float').reshape((28, 28))
	img_bin = binarization(img)
	canvas = np.zeros(shape=(28, 28))
		
	for i in range(len(img_bin)):
		canvas[img_bin[i][0]][img_bin[i][1]] = 1

	return canvas

def get_data(size, isTraining, isRandom):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()

	imgs = False
	labs = False
	idxes = []

	output_imgs = []
	output_labs = []

	if isTraining:
		imgs = images_train
		labs = labels_train
	else:
		imgs = images_test
		labs = labels_test


	if isRandom:
		l = len(labs) - 1

		for i in range(size):
			idxes.append(random.randint(0, l))

	else:
		for i in range(10):
			labs = np.array(labs)
			idx_list = np.where(labs == i)[0].tolist()
			l = len(idx_list) - 1

			for j in range(int(size / 10)):
				idxes.append(idx_list[random.randint(0, l)])

	for i in idxes:

		prep_img = prep_data(imgs[i])
		output_imgs.append(prep_img)
		"""
		output_imgs.append(imgs[i])
		"""

		output_labs.append(labs[i])

	return output_imgs, output_labs



mndata = MNIST('dataset')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()

train_X = np.array(images_train)
test_X = np.array(images_test)

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

train_y_list = []
test_y_list = []

for i in labels_train:
	temp = [0] * 10
	temp[i] = 1
	train_y_list.append(temp)

for i in labels_test:
	temp = [0] * 10
	temp[i] = 1
	test_y_list.append(temp)

train_y = np.array(train_y_list)
test_y = np.array(test_y_list)

print("Train Shape: {}\nTest Shape: {}".format(train_X.shape, test_X.shape))
print("Train Shape: {}\nTest Shape: {}".format(train_y.shape, test_y.shape))



learning_rate = 0.001
n_input = 28
n_classes = 10


"""
epochs = 10
batch_size = 128
"""

batch_size = 25
training_size = batch_size * 1
epochs = 100
testing_size = 50










# ------------------------------------------------------------------------
images_train, labels_train = get_data(10*training_size, True, False)
images_train, labels_train = get_data(10*testing_size, False, False)

train_X = np.array(images_train)
test_X = np.array(images_test)

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

train_y_list = []
test_y_list = []

for i in labels_train:
	temp = [0] * 10
	temp[i] = 1
	train_y_list.append(temp)

for i in labels_test:
	temp = [0] * 10
	temp[i] = 1
	test_y_list.append(temp)

train_y = np.array(train_y_list)
test_y = np.array(test_y_list)
# ------------------------------------------------------------------------
"""
"""








x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, stride = 1):
	x = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k = 2):
	return tf.nn.max_pool(x, ksize = [1, k , k , 1], strides = [1, k , k, 1], padding = "SAME")

weights = {
	'wc1' : tf.get_variable('W0', shape = (3, 3, 1, 32), initializer = tf.contrib.layers.xavier_initializer()),
	'wc2' : tf.get_variable('W1', shape = (3, 3, 32, 64), initializer = tf.contrib.layers.xavier_initializer()),
	'wc3' : tf.get_variable('W2', shape = (3, 3, 64, 128), initializer = tf.contrib.layers.xavier_initializer()),
	'wd1' : tf.get_variable('W3', shape = (4 * 4 * 128, 128), initializer = tf.contrib.layers.xavier_initializer()),
	'out' : tf.get_variable('W4', shape = (128, n_classes), initializer = tf.contrib.layers.xavier_initializer())
}

biases = {
	'bc1': tf.get_variable('B0', shape = (32), initializer = tf.contrib.layers.xavier_initializer()),
	'bc2': tf.get_variable('B1', shape = (64), initializer = tf.contrib.layers.xavier_initializer()),
	'bc3': tf.get_variable('B2', shape = (128), initializer = tf.contrib.layers.xavier_initializer()),
	'bd1': tf.get_variable('B3', shape = (128), initializer = tf.contrib.layers.xavier_initializer()),
	'out': tf.get_variable('B4', shape = (10), initializer = tf.contrib.layers.xavier_initializer()),
}



def conv_net(x, weights, biases):
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k = 2)
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k = 2)
	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
	conv3 = maxpool2d(conv3, k = 2)
	fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init) 
	train_loss = []
	test_loss = []
	train_accuracy = []
	test_accuracy = []
	summary_writer = tf.summary.FileWriter('./Output', sess.graph)
	for i in range(epochs):
		print("-----------------: " + str(epochs) + " - " + str(i))
		for batch in range(len(train_X)//batch_size):
			batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
			batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
			opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
		print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
		print("Optimization Finished!")

		test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
		train_loss.append(loss)
		test_loss.append(valid_loss)
		train_accuracy.append(acc)
		test_accuracy.append(test_acc)
		print("Testing Accuracy:","{:.5f}".format(test_acc))
	summary_writer.close()

"""
plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()


plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

"""
































