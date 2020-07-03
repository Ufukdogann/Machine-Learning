import tensorflow as tf
import numpy as np

learning_rate = 0.001
batch_size = 32
number_of_epochs = 50
training_size = 49000

data = tf.keras.datasets.cifar10.load_data()

train, test = data
train_images, train_labels = train
test_images, test_labels = test

index = np.random.permutation(len(train_images))

training_index = index[:training_size]
validation_index = index[training_size:]

normalised_train_image_pixel = (train_images / 255)
normalised_test_image_pixel = (test_images /255)

training_images = normalised_train_image_pixel[training_index]
training_labels = train_labels[training_index]

validation_images = normalised_train_image_pixel[validation_index]
validation_labels = train_labels[validation_index]

training_label_array, test_label_array, validation_label_array = [], [], []

zero_matrice = np.zeros(10)

for i in range(len(training_labels)):
    number = training_labels[i]
    zero_matrice[number] = 1
    training_label_array.append(zero_matrice)
    zero_matrice = np.zeros(10)

for i in range(len(validation_labels)):
    number = validation_labels[i]
    zero_matrice[number] = 1
    validation_label_array.append(zero_matrice)
    zero_matrice = np.zeros(10)

for i in range(len(test_labels)):
    number = test_labels[i]
    zero_matrice[number] = 1
    test_label_array.append(zero_matrice)
    zero_matrice = np.zeros(10)

training_label_array = np.array(training_label_array).reshape(-1, 10)
test_label_array = np.array(test_label_array).reshape(-1, 10)
validation_label_array = np.array(validation_label_array).reshape(-1, 10)

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32)
Keep_Prob = tf.placeholder(tf.float32)

X_img = tf.reshape(X, [-1, 32, 32, 3])

W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.zeros(shape=(32,)))
A_conv1 = tf.nn.relu(tf.nn.conv2d(X_img, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
b_conv2 = tf.Variable(tf.zeros(shape=(32,)))
A_conv2 = tf.nn.relu(tf.nn.conv2d(A_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

A_pool1 = tf.nn.max_pool(A_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Dropout_1 = tf.nn.dropout(A_pool1, Keep_Prob)

W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
b_conv3 = tf.Variable(tf.zeros(shape=(64,)))
A_conv3 = tf.nn.relu(tf.nn.conv2d(Dropout_1, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
b_conv4 = tf.Variable(tf.zeros(shape=(64,)))
A_conv4 = tf.nn.relu(tf.nn.conv2d(A_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

A_pool2 = tf.nn.max_pool(A_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Dropout_2 = tf.nn.dropout(A_pool2, Keep_Prob)

A_pool2_flat = tf.reshape(Dropout_2, [-1, 8 * 8 * 64])  # ? x 4096

W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 512], stddev=0.1))
b_fc1 = tf.Variable(tf.zeros(shape=(512,)))
A_fc1 = tf.nn.relu(tf.matmul(A_pool2_flat, W_fc1) + b_fc1)  # ? x 1024

Dropout_3 = tf.nn.dropout(A_fc1, Keep_Prob)

W_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.zeros(shape=(10,)))
Z = tf.matmul(Dropout_3, W_fc2) + b_fc2  # ? x 10

# Loss definition
# Important: this function expects weighted inputs, not activations

loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z)
loss = tf.reduce_mean(loss)

hits = tf.equal(tf.argmax(Z, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

# Using Adam instead of gradient descent

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

saver = tf.train.Saver()

session = tf.Session()
session.run(tf.global_variables_initializer())

# Using mini-batches instead of entire dataset

training_index_scalar = len(training_index)
validation_index_scalar = len(validation_index)

for t in range(number_of_epochs):

    indexes_training = np.random.permutation(training_index_scalar)
    indexes_validation = np.random.permutation(validation_index_scalar)

    for i in range(0, training_index_scalar, batch_size):
        images = training_images[indexes_training[i:i + batch_size]]
        labels = training_label_array[indexes_training[i:i + batch_size]]

        session.run(train, {X: images, Y: labels, Keep_Prob: 0.5})

    l,a = session.run([loss,accuracy], {X: validation_images, Y: validation_label_array, Keep_Prob: 1.0})

    print('Epoch: {0}. Validation loss: {1}. Accuracy: {2}.'.format(t, l, a))

saver.save(session, '/tmp/mnist.ckpt')
session.close()

session = tf.Session()
saver.restore(session, '/tmp/mnist.ckpt')

acc = session.run(accuracy, {X: normalised_test_image_pixel, Y: test_label_array, Keep_Prob: 1.0})

print('Test accuracy: {0}.'.format(acc))


