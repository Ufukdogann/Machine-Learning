import numpy as np
import tensorflow as tf
import re
from collections import Counter

def encode_string(str, encoding):
    encode = []

    for character in str:
        encode.append(encoding[character])

    return encode

def decode_string(encode, encoding):
    encoding = {v: k for k, v in encoding.items()}

    string = '{0}'.format(encoding[encode])

    return string

def generate_batches(encoded_text, batch_size, sequence_length):
    block_length = len(encoded_text) // batch_size

    batches = []
    targets = []
    for i in range(0, block_length, sequence_length):
        batch = []
        target = []
        for j in range(batch_size):
            start = j * block_length + i
            end = min(start + sequence_length,
                      j * block_length + block_length)
            batch.append(encoded_text[start:end - 1])
            target.append(encoded_text[start + 1:end])

        batches.append(np.array(batch, dtype=int))
        targets.append(np.array(target, dtype=int))

    return batches, targets

def pick_random_character(char_freq, char_enc):
    character_frequency_values = []
    character_reference_values = []

    for i in char_freq:
        character_frequency_values.append(char_freq[i])

    for j in char_enc:
        character_reference_values.append(char_enc[j])

    random_character = np.random.choice(character_reference_values, p=character_frequency_values)
    return random_character



def main():

    n_epochs = 5
    learning_rate = 0.01
    hidden_units = 256
    batch_size = 16
    sequence_length = 256

    file = open("The_Count_of_Monte_Cristo.txt", encoding='utf-8-sig', mode='r')
    keep_lines = False

    txt = ''

    for line in file.readlines():
        if 'Chapter' not in line or 'VOLUME' not in line or line != '\n':
            subst = ' '
            txt += ''.join(re.sub(r'\n+', subst, line))
    file.close()

    txt = txt.lower()
    txt = re.sub(r'chapter [0-9]+. ', '', txt)
    txt = re.sub(r'[0-9]+m', '', txt)
    txt = re.sub(r'[ ]+', ' ', txt)
    txt = txt.replace('\n', '')

    txt_len = len(txt)
    c_counts = Counter(txt)

    char_freq = {}
    for key, value in c_counts.items():
        char_freq[key] = value / txt_len

    char_enc = {}
    for i, k in enumerate(c_counts.keys()):
        char_enc[k] = i

    number_of_character = len(char_enc)

    inv_dict = {v: k for k, v in char_enc.items()}

    encoded_text = []
    for x in txt:
        encoded_text.append(char_enc[x])

    # Model parameters
    batches, targets = generate_batches(encoded_text, batch_size, sequence_length)
    first_random_character = pick_random_character(char_freq, char_enc)

    # Model definition
    X_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    Y_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    batch_size = tf.shape(X_int)[0]

    # One-hot encoding X_int
    X = tf.one_hot(X_int, depth=number_of_character)  # shape: (batch_size, max_len, k)
    # One-hot encoding Y_int
    Y = tf.one_hot(Y_int, depth=number_of_character)  # shape: (batch_size, max_len, k)

    lstm_1 = tf.contrib.rnn.LSTMCell(num_units=hidden_units, state_is_tuple=True)
    lstm_2 = tf.contrib.rnn.LSTMCell(num_units=hidden_units, state_is_tuple=True)
    multiRNN_cell = tf.contrib.rnn.MultiRNNCell([lstm_1, lstm_2], state_is_tuple=True)

    init_state = multiRNN_cell.zero_state(batch_size, dtype=tf.float32)

    # rnn_outputs shape: (batch_size, max_len, hidden_units)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(multiRNN_cell, X, initial_state=init_state)

    # rnn_outputs_flat shape: ((batch_size * max_len), hidden_units)
    rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

    # Weights and biases for the output layer
    Wout = tf.Variable(tf.truncated_normal(shape=(hidden_units, number_of_character), stddev=0.1))
    bout = tf.Variable(tf.zeros(shape=[number_of_character]))

    # Z shape: ((batch_size * max_len), k)
    Z = tf.matmul(rnn_outputs_flat, Wout) + bout

    Y_flat = tf.reshape(Y, [-1, number_of_character])  # shape: ((batch_size * max_len), k)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y_flat, logits=Z)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for i in range(1, n_epochs + 1):

        init_state_train = multiRNN_cell.zero_state(batch_size, dtype=tf.float32)
        prev_state = session.run([init_state_train], {X_int: batches[0], Y_int: targets[0]})

        for i in range(1, len(batches)):
            feed = {X_int: batches[i], Y_int: targets[i], init_state: prev_state}
            l, _, prev_state = session.run([s, train, final_state], feed)

            print('Epoch: {0}. Loss: {1}.'.format(i, l))

    zmulti = tf.multinomial(Z, 1)

    saver.save(session, './saved_2/model.ckpt')

    init_state_prediction = multiRNN_cell.zero_state(batch_size, dtype=tf.float32)

    prev_state = None

    res = ""
    for t in range(sequence_length):
        ch = first_random_character
        if prev_state is None:
            prev_state = session.run([init_state_prediction], {X_int: [[ch]]})

        feed = {X_int: [[ch]], init_state: prev_state}
        zpred, guess, prev_state = session.run([Z, zmulti, final_state], feed)
        ch = guess[0][0]
        res += inv_dict[ch]

    print(res)


if __name__ == "__main__":
    main();
#Ufuk Dogan
