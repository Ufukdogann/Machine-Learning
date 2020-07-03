import numpy as np
import tensorflow as tf
import os
import unicodedata
import re
from tensorflow.contrib import layers


def download_the_translation_set():

    path_to_zip = tf.keras.utils.get_file("spa-eng.zip", origin="http://download.tensorflow.org/data/spa-eng.zip",
                                          extract=True)
    path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

    file = open(path_to_file, encoding='utf8', mode='r')
    text = file.read().lower()

    return text


def creating_the_english_and_spanish_set(text):

    limit = 30000

    text_array = text.split("\n")
    text_array = text_array[:limit]

    np.random.shuffle(text_array)

    for i in range(0, len(text_array)):
        text_array[i] = re.sub(r"([?.!,¿¡])", r" \1 ", text_array[i])
        text_array[i] = re.sub(r'[" "]+', " ", text_array[i])

    english_set, spanish_set = [], []

    for line in text_array:

        start_token = "<START> "
        end_token = " <END>"

        splitted_line = line.split("\t")
        english_line = start_token + splitted_line[0] + end_token
        spanish_line = start_token + splitted_line[1] + end_token

        english_line_array = english_line.split()
        spanish_line__array = spanish_line.split()

        english_line_array_without_whitespace = list(filter(None, english_line_array))
        spanish_line_array_without_whitespace = list(filter(None, spanish_line__array))

        english_set.append(english_line_array_without_whitespace)
        spanish_set.append(spanish_line_array_without_whitespace)

    return english_set, spanish_set


def unicode_to_ascii(s):

    return ''.join(c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn")


def word_to_int(english_set, spanish_set):

    english_wordInt_dictionary, spanish_wordInt_dictionary = {}, {}
    english_wordInt_dictionary["<pad>"] = 0
    spanish_wordInt_dictionary["<pad>"] = 0

    english_word_counter, spanish_word_counter = 0, 0

    for i in range(len(english_set)):
        for english_word in english_set[i]:
            if english_word not in english_wordInt_dictionary.keys():
                english_wordInt_dictionary[english_word] = english_word_counter + 1
                english_word_counter = english_word_counter + 1

    for i in range(len(spanish_set)):
        for spanish_word in spanish_set[i]:
            if spanish_word not in spanish_wordInt_dictionary.keys():
                spanish_wordInt_dictionary[spanish_word] = spanish_word_counter + 1
                spanish_word_counter = spanish_word_counter + 1

    return english_wordInt_dictionary, spanish_wordInt_dictionary

def decode(lines, dictionary):
    lines = lines.tolist()

    for i in range(len(lines)):
        traduce_flag = True
        for jj in range(len(lines[i])):
            if traduce_flag:
                lines[i][jj] = dictionary[lines[i][jj]]
                if lines[i][jj] == '<EOS>':
                    traduce_flag = False
                    rem_index = jj + 1
            else:
                del lines[i][rem_index]

    return lines



def int_to_word(english_wordInt_dictionary, spanish_wordInt_dictionary):

    english_intWord_dictionary = dict((v, k) for k, v in english_wordInt_dictionary.items())
    spanish_intWord_dictionary = dict((v, k) for k, v in spanish_wordInt_dictionary.items())

    return english_intWord_dictionary, spanish_intWord_dictionary


def train(english_set, spanish_set, english_wordInt_dictionary, spanish_wordInt_dictionary):

    english_dataset, spanish_dataset, english_converted_to_int_sentences, spanish_converted_to_int_sentences, length_of_english_sentences, length_of_spanish_sentences = [], [], \
                                                                                                               [], [], [], []

    for i in range(len(english_set)):
        newArray_english = []
        for english_word in english_set[i]:
            newArray_english.append(english_wordInt_dictionary[english_word])
        english_dataset.append(newArray_english)
        length_of_english_sentences.append(len(english_set[i]))


    for i in range(len(spanish_set)):
        newArray_spanish = []
        for spanish_word in spanish_set[i]:
            newArray_spanish.append(spanish_wordInt_dictionary[spanish_word])
        spanish_dataset.append(newArray_spanish)
        length_of_spanish_sentences.append(len(spanish_set[i]))


    max_length_english = max([len(t) for t in english_dataset])
    max_length_spanish = max([len(t) for t in spanish_dataset])

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(spanish_dataset,
                                                                 maxlen=max_length_spanish,
                                                                 padding='post')

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(english_dataset,
                                                                  maxlen=max_length_english,
                                                                  padding='post')
    vocabulary_size_english = len(english_wordInt_dictionary)
    vocabulary_size_spanish = len(spanish_wordInt_dictionary)

    limit = 30000
    training_set_limit = int(limit * 0.6)
    validation_set_limit = int(limit * 0.1)
    test_set_limit = int(limit * 0.3)
    batch_size = 64
    buffer_size = 10
    embedding_size = 256
    hidden_size = 1024
    learning_rate = 0.001

    training_set_english = target_tensor[:training_set_limit]
    validation_set_english = target_tensor[training_set_limit:training_set_limit + validation_set_limit]
    test_set_english = target_tensor[training_set_limit + validation_set_limit:training_set_limit + validation_set_limit + test_set_limit]

    training_set_spanish = input_tensor[:training_set_limit]
    validation_set_spanish = input_tensor[training_set_limit:training_set_limit + validation_set_limit]
    test_set_spanish = input_tensor[training_set_limit + validation_set_limit:training_set_limit +
                                                                                validation_set_limit + test_set_limit]

    input_sentences_placeholder = tf.placeholder(shape=[None, None], dtype=tf.int32)
    input_sentenes_lengths_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)

    target_sentences_placeholder = tf.placeholder(shape=[None, None], dtype=tf.int32)
    target_sentences_lengths_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((training_set_spanish,
                                                        training_set_english,
                                                        length_of_english_sentences[:training_set_limit]
                                                        )).\
        shuffle(buffer_size).\
        batch(batch_size,
              drop_remainder=True).\
        repeat(buffer_size)

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_set_spanish,
                                                             validation_set_english,
                                                             length_of_english_sentences[training_set_limit:training_set_limit + validation_set_limit]
                                                             ))\
        .shuffle(buffer_size).\
        batch(batch_size, drop_remainder=True).\
        repeat(buffer_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_set_spanish,
                                                       test_set_english))\
        .shuffle(buffer_size)\
        .batch(batch_size,
               drop_remainder=True)\
        .repeat(buffer_size)

    dynamic_batch_size = tf.shape(input_sentences_placeholder)[0]

    word_embeddings_spanish = tf.get_variable("word_embeddings_spanish",
                                              shape=[vocabulary_size_spanish, embedding_size])

    embedded_input_sentences_spanish_training = tf.nn.embedding_lookup(word_embeddings_spanish,
                                                                       input_sentences_placeholder)

    word_embeddings_english = tf.get_variable("word_embeddings_english",
                                              shape=[vocabulary_size_english, embedding_size])

    embedded_target_sentences_english_training = tf.nn.embedding_lookup(word_embeddings_english,
                                                                        target_sentences_placeholder)

    encoder_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=hidden_size)

    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, embedded_input_sentences_spanish_training,
                                                             dtype=tf.float32)

    decoder_input = embedded_target_sentences_english_training[:, :-1]

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hidden_size, encoder_outputs)
    decoder_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=hidden_size)
    attention_wrapper = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, alignment_history=True)

    decoder_init_state = attention_wrapper.zero_state(dynamic_batch_size, tf.float32).clone(cell_state=encoder_final_state)

    training_decoder_helper = tf.contrib.seq2seq.TrainingHelper(decoder_input,
                                                                target_sentences_lengths_placeholder - 1)
    projection_layer = tf.layers.Dense(vocabulary_size_english)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_wrapper,
                                                       helper=training_decoder_helper,
                                                       initial_state=decoder_init_state,
                                                       output_layer=projection_layer)

    decoder_outputs, decoder_final_state, decoder_final_sequence_length = tf.contrib.seq2seq.dynamic_decode(training_decoder)
    training_logits = decoder_outputs.rnn_output
    train_predictions = decoder_outputs.sample_id

    stack_logits_tensors = tf.stack([tf.shape(target_sentences_placeholder)[1], tf.shape(training_logits)[1]])
    maximum_logit_tensor = stack_logits_tensors[tf.argmax(stack_logits_tensors)]
    logit_difference = maximum_logit_tensor - tf.shape(training_logits)[1]

    padded_training_logits = tf.pad(tensor=training_logits,
                               paddings=[[0, 0], [0, logit_difference], [0, 0]])

    training_mask = tf.sequence_mask(lengths=target_sentences_lengths_placeholder - 1,
                            dtype=tf.float32)

    mask_difference = maximum_logit_tensor - tf.shape(training_mask)[1]

    training_padded_mask = tf.pad(tensor=training_mask,
                         paddings=[[0, 0], [0, mask_difference]])

    training_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_sentences_placeholder,
                                                                   logits=padded_training_logits)

    training_loss = tf.reduce_sum((training_loss*training_padded_mask))/tf.reduce_sum(training_padded_mask)

    inference_decoder_input = word_embeddings_english

    inf_decoder_init_state = attention_wrapper.zero_state(batch_size=dynamic_batch_size, dtype=tf.float32).clone(
        cell_state=encoder_final_state)

    inference_decoder_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(inference_decoder_input,
                                                                        tf.fill([dynamic_batch_size],
                                                                        english_wordInt_dictionary["<START>"]),
                                                                        english_wordInt_dictionary["<END>"])

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_wrapper,
                                                       helper=inference_decoder_helper,
                                                       initial_state=inf_decoder_init_state,
                                                       output_layer=projection_layer)

    inference_decoder_max_iterations = tf.round(tf.reduce_max(max_length_english) * 2)

    inference_decoder_outputs, inference_decoder_final_state, inference_decoder_final_sequence_length = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                                                                                          maximum_iterations=inference_decoder_max_iterations,
                                                                                                                                          )

    inference_logits = inference_decoder_outputs.rnn_output
    inf_predictions = inference_decoder_outputs.sample_id

    stack_inference_logit_tensors = tf.stack([tf.shape(target_sentences_placeholder)[1], tf.shape(inference_logits)[1]])
    maximum_inference_logits_tensor = stack_inference_logit_tensors[tf.argmax(stack_inference_logit_tensors)]
    inference_logits_difference = maximum_inference_logits_tensor - tf.shape(inference_logits)[1]

    padded_inference_logits = tf.pad(tensor=inference_logits,
                                     paddings=[[0, 0], [0, inference_logits_difference], [0, 0]])

    inference_mask = tf.sequence_mask(lengths=target_sentences_lengths_placeholder - 1,
                                      dtype=tf.float32)

    mask_inference_difference = maximum_inference_logits_tensor - tf.shape(inference_mask)[1]

    inference_padded_mask = tf.pad(tensor=inference_mask,
                                   paddings=[[0, 0], [0, mask_inference_difference]])

    placeholder_difference = maximum_inference_logits_tensor - tf.shape(target_sentences_placeholder)[1]

    placeholder_padded = tf.pad(tensor=target_sentences_placeholder,
                                paddings=[[0, 0], [0, placeholder_difference]])

    inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=placeholder_padded,
                                                                    logits=padded_inference_logits)

    inference_loss = tf.reduce_sum((inference_loss*inference_padded_mask))/tf.reduce_sum(inference_padded_mask)

    argmax = inference_decoder_outputs.sample_id

    saver = tf.train.Saver()
    session = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(training_loss)

    session.run(tf.global_variables_initializer())

    iteration_number = 5000
    n_val = 3000

    training_iterator = train_dataset.make_one_shot_iterator()
    training_next_batch = training_iterator.get_next()

    for i in range(iteration_number):

        (X_training, Y_training, Y_length_training) = session.run(training_next_batch)

        feed_train = {
                input_sentences_placeholder: X_training,
                target_sentences_placeholder: Y_training,
                target_sentences_lengths_placeholder: Y_length_training
                      }

        train_loss, _, = session.run([training_loss, train], feed_train)

        if i == 2801:
            print('Iteration: {0}. Training Loss: {1}.'.format(i+1, train_loss))
            predicted_sentences = session.run([train_predictions], feed_train)
            words = decode(predicted_sentences[0], english_intWord_dictionary)

            prediction_targets = session.run([target_sentences_placeholder], feed_train)
            predicted_word = decode(prediction_targets[0], english_intWord_dictionary)
            for j in range(len(words)):
                print(words[j])
                print(predicted_word[j])
            
             n_steps_val = int(np.round(n_val / 64))
             for ev in range(1, n_steps_val + 1):
            
                 feed_validation = {
                         input_sentences_placeholder: validation_set_spanish,
                         target_sentences_placeholder: validation_set_english,
                         target_sentences_lengths_placeholder: length_of_english_sentences[training_set_limit:
                                                                                           training_set_limit + validation_set_limit]
                                    }

                 validation_loss = session.run([training_loss], feed_validation)
             print('Iteration: {0}. Validation Loss: {1}.'.format(i + 1, validation_loss))


text = download_the_translation_set()
s = unicode_to_ascii(text)
english_set, spanish_set = creating_the_english_and_spanish_set(s)
english_wordInt_dictionary, spanish_wordInt_dictionary = word_to_int(english_set, spanish_set)
english_intWord_dictionary, spanish_intWord_dictionary = int_to_word(english_wordInt_dictionary,
                                                                     spanish_wordInt_dictionary)
train(english_set, spanish_set, english_wordInt_dictionary, spanish_wordInt_dictionary)






