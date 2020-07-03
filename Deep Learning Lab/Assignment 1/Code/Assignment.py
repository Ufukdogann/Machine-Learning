import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mp

def create_dataset(w_star, x_range, sample_size, sigma, seed = None):
    random_state = np.random.RandomState(seed)

    x = random_state.uniform(x_range[0], x_range[1], (sample_size))
    X = np.zeros((sample_size, w_star.shape[0]))

    for i in range(sample_size):
        X[i, 0] = 1.
        for j in range(1, w_star.shape[0]):
            X[i, j] = x[i]**j

    y = X.dot(w_star)
    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size)

    return X, y, x

def equation(x,weight):

    result = weight[3] * x ** 3 + weight[2] * x ** 2 + weight[1] * x + weight[0]
    return result


def main():

    x_range = [-3, 2]
    sample_size_train = 100
    sample_size_value = 100
    sigma = 0.5

    weight = np.array([-8, -4, 2, 1])

    n_dimensions = weight.shape[0]
    n_iterations = 5000
    learning_rate = 0.005

    # Placeholder for the data matrix, where each observation is a row
    X = tf.placeholder(tf.float32, shape=(None, n_dimensions))
    # Placeholder for the targets
    y = tf.placeholder(tf.float32, shape=(None,))
    # Variable for the model parameters
    w = tf.Variable(tf.zeros((n_dimensions, 1)), trainable=True)

    # Loss function
    prediction = tf.reshape(tf.matmul(X, w), (-1,))
    loss = tf.reduce_mean(tf.square(y - prediction))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)  # Gradient descent update operation

    initializer = tf.global_variables_initializer()

    X_train, y_train, x_train = create_dataset(weight, x_range, sample_size_train, sigma, 0)

    session = tf.Session()
    session.run(initializer)

    for t in range(1, n_iterations + 1):
        l, _ = session.run([loss, train], feed_dict={X: X_train, y: y_train})
        print('Iteration {0}. Loss: {1}.'.format(t, l))

    X_val, y_val, x_val = create_dataset(weight, x_range, sample_size_value, sigma, 1)
    l = session.run(loss, feed_dict={X: X_val, y: y_val})
    print('Validation loss: {0}.'.format(l))

    print(session.run(w).reshape(-1))

    x_coord = [x for x in np.arange(-3, 2, 0.01)]
    y_coord_star = [equation(x, weight) for x in x_coord]
    y_coord_cap = [equation(x, session.run(w)) for x in x_coord]

    mp.scatter(x_val, y_val, c='b', s=4, label='validation set')
    mp.scatter(x_train, y_train, c='r', s=4, label='training set')
    mp.plot(x_coord, y_coord_cap, c='y', label='w^')
    mp.plot(x_coord, y_coord_star, c='k', label='w*')
    mp.legend()
    mp.show()

    session.close()

if __name__ == "__main__":
    main()