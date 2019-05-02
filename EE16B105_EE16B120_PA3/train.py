# -*- coding: utf-8 -*-
"""Transilteration2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gXFxbQj_k5s7jzhxKBmnNY83oQr3rVAt
"""

# Note : We have referenced https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f for our code

import tensorflow as tf
import matplotlib.pyplot as plot
import pickle 
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorflow.python.framework import ops
import os

def get_specs() :
    "Function to get the specifications from the command line arguments"
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",type=float,help="initial learning rate of the algorithm")
    parser.add_argument("--batch_size",type=int,help="size of each mini-batch")
    parser.add_argument("--init",type=int,help="type of parameter initialization")
    parser.add_argument("--dropout_prob",type=float)
    parser.add_argument("--decode_method",type=int)
    parser.add_argument("--beam_width",type=float)
    parser.add_argument("--epochs",type=int,help="number of time steps for which we train our model")
    parser.add_argument("--save_dir",help="the directory where pickled model should be saved")
    parser.add_argument("--train",help="path to the training dataset")
    parser.add_argument("--val",help="path to the validation dataset")
    parser.add_argument("--test",help="path to the test dataset")
    
    args = parser.parse_args()
    return args

def encoder(rnn_inputs, rnn_size, keep_prob, source_vocab_size, encoding_embedding_size,init):
    
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, vocab_size=source_vocab_size, embed_dim=encoding_embedding_size)
    
    # Bi-directional LSTM
    if init == 1 :
      lstm_fw = tf.contrib.rnn.LSTMCell(int(rnn_size/2),initializer=tf.contrib.layers.xavier_initializer())
      lstm_bw = tf.contrib.rnn.LSTMCell(int(rnn_size/2),initializer=tf.contrib.layers.xavier_initializer())
    
      lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw,keep_prob)
      lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw,keep_prob)
    
      outputs,state = tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,embed,dtype=tf.float32)
      outputs = tf.concat(outputs,2)
    
      # Single LSTM Layer
      #lstm = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.contrib.layers.xavier_initializer())
      #lstm = tf.contrib.rnn.DropoutWrapper(lstm,keep_prob)

      #outputs,state = tf.nn.dynamic_rnn(lstm,embed,dtype=tf.float32)
    
    else :
      lstm_fw = tf.contrib.rnn.LSTMCell(int(rnn_size/2),initializer=tf.random_uniform_initializer(-0.1,0.1))
      lstm_bw = tf.contrib.rnn.LSTMCell(int(rnn_size/2),initializer=tf.random_uniform_initializer(-0.1,0.1))
    
      lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw,keep_prob)
      lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw,keep_prob)
    
      outputs,state = tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,embed,dtype=tf.float32)
      outputs = tf.concat(outputs,2)
    
      # Single LSTM Layer
      #lstm = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1))
      #lstm = tf.contrib.rnn.DropoutWrapper(lstm,keep_prob)

      #outputs,state = tf.nn.dynamic_rnn(lstm,embed,dtype=tf.float32)
    
    return outputs, state

def decoder(dec_input, encoder_state,enc_outputs,target_sequence_length, max_target_sequence_length, rnn_size,
            num_layers, target_vocab_to_int, target_vocab_size,
            batch_size, keep_prob, decoding_embedding_size,init):
    # Function : To implement the decoder training and decoder inference part
    # Parameters : dec_input - Input to the decoder(Process the actual input values(characters in Hindi))
    #              encoder_state : Final state of the encoder
    #              enc_outputs : Outputs of the encoders which will pass through the attention mechanism
    #              target_sequence_length : The length of each target Hindi word
    #              max_target_sequence_length : The maximum length value in the target sequence length
    #              rnn_size : The size of the decoder RNN layer
    #              num_layers : Number of layers in the RNN decoder output
    
    target_vocab_size = len(target_vocab_to_int) # Size of the Hindi vocabulary
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size,enc_outputs)                                         
    
    if init==1 :
      cell_list = []
      for i in range(num_layers) :
        single_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.contrib.layers.xavier_initializer())
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell,keep_prob)
        cell_list.append(single_cell)
      if num_layers == 1 :
        cells = cell_list[0]
      else :
        cells = tf.contrib.rnn.MultiRNNCell(cell_list)
    else :
      cell_list = []
      for i in range(num_layers) :
        single_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1))
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell,keep_prob)
        cell_list.append(single_cell)
      if num_layers == 1 :
        cells = cell_list[0]
      else :
        cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        
    cells = tf.contrib.seq2seq.AttentionWrapper(cells,attention_mechanism,rnn_size,alignment_history=True) # alignment_history set true to observe the attention weights
    
    dec_init = cells.zero_state(batch_size,dtype=tf.float32)
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
        decoder = tf.contrib.seq2seq.BasicDecoder(cells, helper, dec_init, output_layer)
        train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)
        
    with tf.variable_scope("decode", reuse=True):
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,tf.fill([batch_size], target_vocab_to_int['<GO>']),target_vocab_to_int['<EOW>'])
        decoder = tf.contrib.seq2seq.BasicDecoder(cells, helper, dec_init, output_layer)
        infer_output, infer_states, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)

    return (train_output, infer_output, infer_states)

def transliterate(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int,init):
   # Function to implement the end-to-end sequence to sequence model
    
    enc_outputs, enc_states = encoder(input_data, rnn_size, keep_prob, source_vocab_size, enc_embedding_size, init)
    
    # We need to modify the inputs so that the initial character for each translated word is a <GO> character
    go_id = target_vocab_to_int['<GO>']
    
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    train_output, infer_output, infer_states = decoder(dec_input, enc_states,enc_outputs, target_sequence_length, 
                                               max_target_sentence_length,rnn_size,num_layers,target_vocab_to_int,
                                              target_vocab_size,batch_size,keep_prob,dec_embedding_size,init)
    
    return train_output, infer_output, infer_states

def get_data(train_path,val_path,test_path) :
  # Function - To get data(.csv) files from their location in the drive
  # Parameters - train_path : Location of the train data
  #              val_path : Location of the validation data
  #              test_path : Location of the test data
  
  tf.reset_default_graph()
  
  train_data = pd.read_csv(train_path)
  val_data = pd.read_csv(val_path)
  train = np.array(train_data)
  val = np.array(val_data)
  
  # Find the unique characters to determine the vocabulary of the input and output
  chars = []
  chars_h = []
  max_len = 0
  max_len_o = 0
  word = 0
  word_o = 0
  
  # Pre-processing for the train data
  
  for i in range(train.shape[0]) :
    train[i,1] = train[i,1].replace(" ","")
    train[i,2] = train[i,2].replace(" ","")
    #train[i,2] = '$'+train[i,2] # '$' is the <GO> character
  
  # Find all the unique characters in the inpu data-set
  for i in range(train.shape[0]) :
    for j in range(len(train[i,1])) :
      chars.append(train[i][1][j])
    for k in range(len(train[i,2])) :
      chars_h.append(train[i][2][k])
  u_set = set(chars) # Set stores only unique characters
  u_list = (list(u_set))
  u_list.sort()
  print(u_list)
  Vt_in = len(u_list) # Size of vocabulary of the input characters
  uh_set = set(chars_h)
  uh_list = (list(uh_set))
  uh_list.sort()
  print(uh_list)
  Vt_out = len(uh_list) # Size of vocabulary of the output characters
  
  inp_voc = {}
  out_voc = {}
  inp_voc["<PAD>"] = 0
  inp_voc["<EOW>"] = 1
  inp_voc["<UNK>"] = 2
  inp_voc["<GO>"] = 3
  out_voc["<PAD>"] = 0
  out_voc["<EOW>"] = 1
  out_voc["<UNK>"] = 2
  out_voc["<GO>"] = 3
  i = 4
  for ch in u_list :
    inp_voc[ch] = i
    i = i + 1
  Vt_in = Vt_in + 4
  print(Vt_in)
  inp_voc_int = {v_i:v for v,v_i in inp_voc.items()}
  
  i = 4
  for ch in uh_list :
    out_voc[ch] = i
    i = i + 1
  out_voc_int = {v_i:v for v,v_i in out_voc.items()}
  Vt_out = Vt_out + 4
  print(Vt_out)
  # Created dictionaries with the vocab of the input and output
  
  ids_i = []
  ids_o = []
  
  for i in range(train.shape[0]) :
    temp = []
    for ch in train[i,1] :
      temp.append(inp_voc[ch])
    ids_i.append(temp)
    temp_val = []
    for ch in train[i,2] :
      temp_val.append(out_voc[ch])
    temp_val.append(out_voc["<EOW>"])
    ids_o.append(temp_val)
  
  # Pre-processing for the validation data
 
  for i in range(val.shape[0]) :
    val[i,1] = val[i,1].replace(" ","")
    val[i,2] = val[i,2].replace(" ","")
  
  ids_i_val = []
  ids_o_val = []
  
  for i in range(val.shape[0]) :
    temp = []
    for ch in val[i,1] :
      temp.append(inp_voc[ch])
    ids_i_val.append(temp)
    temp_val = []
    for ch in val[i,2] :
      temp_val.append(out_voc[ch])
    temp_val.append(out_voc["<EOW>"])
    ids_o_val.append(temp_val)
  
  # Pre-processing for the test data
  #max_len_test = 0
  #for i in range(test.shape[0]) :
  #  test[i,1] = test[i,1].replace(" ","")
  
  ids_i_test = []
  if test_path is not None :
    test_data = pd.read_csv(test_path)
    test = np.array(test_data)
    for i in range(test.shape[0]) :
    	test[i,1] = test[i,1].replace(" ","")
    for i in range(test.shape[0]) :
      temp = []
      for ch in test[i,1] :
        if ch in inp_voc :
          temp.append(inp_voc[ch])
        else :
          temp.append(inp_voc["<UNK>"])
      ids_i_test.append(temp)
  
  return ids_i, ids_o,ids_i_val, ids_o_val, ids_i_test, inp_voc, out_voc, inp_voc_int, out_voc_int

# Neural Network Parameters : will obtain them from the command line arguments
#learning_rate = 7e-4
#keep_probability=0.7
#batch_size = 15
#epochs = 20
num_layers = 2
rnn_size = 512
encoding_embedding_size = 256
decoding_embedding_size = 256
#display_step = 10
#init = 1
#save_dir = ""
train_losses = []
val_losses = []

args = get_specs()
learning_rate = args.lr
batch_size = args.batch_size
dropout_prob = args.dropout_prob
keep_probability = 1 - dropout_prob
init = args.init
epochs = args.epochs
save_dir = args.save_dir
train_path = args.train
val_path = args.val
test_path = args.test

train_source, train_target, valid_source, valid_target, test_source, source_vocab_to_int, target_vocab_to_int, inp_voc_int, out_voc_int = get_data(train_path,val_path,test_path)
max_target_sentence_length = max([len(word) for word in train_source])

def pad_tensor(infer_logits_train,targets) :
  temp1 = tf.shape(targets)[0]
  temp2 = tf.shape(infer_logits_train)[1]
  targets_infer = tf.slice(targets,[0,0],[temp1,temp2])
  return infer_logits_train,targets_infer

def pad_target(infer_logits_train,targets) :
  temp1 = tf.shape(infer_logits_train)[0]
  temp2 = tf.shape(targets)[1]
  temp3 = tf.shape(infer_logits_train)[2]
  infer_logits_train2 = tf.slice(infer_logits_train,[0,0,0],[temp1,temp2,temp3])
  return infer_logits_train,targets

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

def get_accuracy(logits,target):
    acc =0 
    for i in range(len(logits)) :
      pred = []
      for k in range(len(logits[i])) :
        if logits[i][k] == source_vocab_to_int['<EOW>'] :
          break
        else :
          pred.append(logits[i][k])
      act = []
      for k in range(len(target[i])) :
        if target[i][k] == source_vocab_to_int['<EOW>'] :
          break
        else :
          act.append(target[i][k])
      # Compare the two obtained words
      if len(pred) == len(act) :
        count = 0
        for m in range(len(pred)) :
          if pred[m] == act[m] :
            count = count + 1
          else :
            break
        if count == len(pred) :
          acc = acc + 1
    return acc

def train(train_source,train_target,valid_source,valid_target,test_source,source_vocab_to_int,target_vocab_to_int,inp_voc_int,out_voc_int,
          learning_rate, keep_probability, batch_size , epochs, num_layers, rnn_size, encoding_embedding_size, decoding_embedding_size,init,save_dir) :
  
  train_graph = tf.Graph()
  with train_graph.as_default():
      input_data = tf.placeholder(tf.int32, [None, None], name='input')
      targets = tf.placeholder(tf.int32, [None, None], name='targets')
    
      target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
      max_target_sequence_length = tf.reduce_max(target_sequence_length) 
    
      lr = tf.placeholder(tf.float32,name='lr_rate')
      keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    
      train_logits, inference_logits, infer_states = transliterate(tf.reverse(input_data, [-1]),
                                                    targets,
                                                    keep_prob,
                                                    batch_size,
                                                    target_sequence_length,
                                                    max_target_sequence_length,
                                                    len(source_vocab_to_int),
                                                    len(target_vocab_to_int),
                                                    encoding_embedding_size,
                                                    decoding_embedding_size,
                                                    rnn_size,
                                                    num_layers,
                                                    target_vocab_to_int,init)
    
      training_logits = tf.identity(train_logits.rnn_output, name='logits')
      infer_logits_train = tf.identity(inference_logits.rnn_output, name='logits_infer')
      inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
      infer_states = tf.identity(infer_states.alignment_history.stack(),name='attention_weights')
    
    # - Returns a mask tensor representing the first N positions of each cell.
      masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

      infer_logits_train,targets_infer = tf.cond(tf.shape(infer_logits_train)[1] < tf.shape(targets)[1],lambda :pad_tensor(infer_logits_train,targets),
                 lambda :pad_target(infer_logits_train,targets))
    
      #masks_infer = tf.cond(tf.shape(targets_infer)[1] < tf.shape(targets)[1],lambda :new_mask(target_sequence_length,targets_infer))
    
      with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
          cost = tf.contrib.seq2seq.sequence_loss(training_logits,targets,masks)
          cost_infer = tf.contrib.seq2seq.sequence_loss(infer_logits_train,targets_infer,masks)
          
          # Optimizer
          optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
          gradients = optimizer.compute_gradients(cost)
          clip = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
          train_op = optimizer.apply_gradients(clip)
      
  patience = 5
  end_epoch = 0
  i = 0
  val_count = 0
  val_costs = []
  val_costs_plot = []
  train_costs_plot = []
  if not os.path.exists(save_dir) :
    os.makedirs(save_dir)

  with tf.Session(graph=train_graph) as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
    
      for epoch in range(epochs):
        
        batch_loss = 0.0
        val_loss = 0.0
        acc = 0.0
        
        for batch, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss / (len(train_source) // batch_size)
  
        for batch_val, (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) in enumerate(
        get_batches(valid_source, valid_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):
                  
          val_loss_temp, val_logits = sess.run([cost,inference_logits],feed_dict={input_data:valid_sources_batch,targets:valid_targets_batch,target_sequence_length: valid_targets_lengths,
                                                   keep_prob:1.0})
          val_loss += val_loss_temp / (len(valid_source) // batch_size)
          acc += get_accuracy(val_logits,valid_targets_batch) / (batch_size * (len(valid_source) // batch_size))
        
        val_costs_plot.append(val_loss)
        train_costs_plot.append(batch_loss)
        
        if i>=1 :
          if val_loss > val_costs[i-1] :
            val_count = val_count + 1
          else :
            val_count = 0
          val_costs.append(val_loss)
        else :
          val_costs.append(val_loss)
        i = i + 1
        
        if val_count == patience :
          end_epoch = epoch - patience
          saver.restore(sess,save_dir+"weights-"+str(end_epoch)+".ckpt")
          break       
        
        if epoch % 7 == 0 :
          learning_rate = learning_rate / 2
        
        saver.save(sess,save_dir+"weights-"+str(epoch)+".ckpt")
        
        print("Epoch : "+str(epoch)+"Batch : "+str(batch)+"Training Loss : " + str(batch_loss))
        print("Epoch : "+str(epoch)+"Batch : "+str(batch)+"Validation Loss : " + str(val_loss))
        print("Epoch : "+str(epoch)+"Batch : "+str(batch)+"Validation Accuracy : " + str(acc))
        
      if end_epoch == 0 :
        end_epoch = epoch
      
  return train_costs_plot, val_costs_plot, end_epoch

train_loss, val_loss, end_epoch = train(train_source,train_target,valid_source,valid_target,test_source,source_vocab_to_int,target_vocab_to_int,inp_voc_int,out_voc_int, learning_rate, keep_probability, batch_size , epochs, num_layers, rnn_size, encoding_embedding_size, decoding_embedding_size,init,save_dir)
train_losses.append(train_loss)
val_losses.append(val_loss)

def visualize(end_epoch) :
  # Function to obtain a translation of a given word and observe the attention weights
  loaded_graph = tf.Graph()
  
  #https://drive.google.com/file/d/1lvOsUVCMuGnBrNg7F3bMuD5podIU3Xun/view?usp=sharing
  file_id = '1lvOsUVCMuGnBrNg7F3bMuD5podIU3Xun'
  train_temp = drive.CreateFile({'id': file_id})
  train_temp.GetContentFile("Nirmala.ttf")
  
  with tf.Session(graph=loaded_graph) as sess :
    loader = tf.train.import_meta_graph('weights-'+str(end_epoch)+'.ckpt.meta')
    loader.restore(sess,'weights-'+str(end_epoch)+'.ckpt')
    
    data = test_source[0:batch_size]
    
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    infer_states = loaded_graph.get_tensor_by_name('attention_weights:0') # These are the attention weights
    
    pad_sources_batch = np.array(pad_sentence_batch(data, source_vocab_to_int['<PAD>']))
    pad_source_lengths = []
    for source in pad_sources_batch:
      pad_source_lengths.append(len(source)) 
    
    alignments, logs = sess.run([infer_states,logits],feed_dict={input_data : pad_sources_batch,target_sequence_length :pad_source_lengths,keep_prob:1.0})
    print(alignments.shape)
    
    one_align = np.reshape(alignments[:,5,:],(alignments.shape[0],alignments.shape[2])).T
    print(one_align.shape)
    
    word_pair = pad_sources_batch[5]
    print(word_pair.shape)
    Y_label = []
    for i in range(len(word_pair)) :
      if word_pair[i] == 0 :
        break
      Y_label.append(inp_voc_int[word_pair[i]])
    print(Y_label)
    
    target = logs[5]
    print(target.shape)
    X_label = []
    for k in range(len(target)) :
      if target[k] == 1 :
        break
      X_label.append(out_voc_int[target[k]])
    print(X_label)
    one_align = one_align[0:len(Y_label),0:len(X_label)]
    
    fontP = font_manager.FontProperties(fname = 'Nirmala.ttf')
    #fontP = font_manager.FontProperties(fname = 'AksharUnicode.ttf')
    #Use one of the two .ttf files above.
    #Note that the .ttf file should be in the same place as this code 
    #or specify the correct path to the file
    
    fontP.set_size(16)
    
    #Rest of the plotting code below
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(5, 4))  # set figure size
    heatmap = ax.pcolor(one_align, cmap=plt.cm.Blues, alpha=0.9)

    if X_label != None and Y_label != None:
        #decode fn used below takes care of making labels/Hindi chars unicode
        #X_label = [x_label.decode('utf-8') for x_label in X_label]
        #Y_label = [y_label.decode('utf-8') for y_label in Y_label]

        xticks = range(0, len(X_label))
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels('')
        
        xticks1 = [k+0.5 for k in xticks]
        ax.set_xticks(xticks1, minor=True)
        ax.set_xticklabels(X_label, minor=True, fontproperties=fontP)  # labels should be 'unicode'
        #using fontP from above to get Hindi chars

        yticks = range(0, len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels('')
        
        yticks1 = [k+0.5 for k in yticks]
        ax.set_yticks(yticks1, minor=True)
        ax.set_yticklabels(Y_label, minor=True,fontproperties=fontP)  # labels should be 'unicode'

        ax.grid(True)
    plt.title(u'Attention Heatmap')
    plt.show()

#visualize(end_epoch=19)

def predict(end_epoch) :

  loaded_graph = tf.Graph()
  with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
      loader = tf.train.import_meta_graph('weights-'+str(end_epoch)+'.ckpt.meta')
      loader.restore(sess, "weights-"+str(end_epoch)+".ckpt")

      # Append random indices to test_source so that it becomes a multiple of batch_size
      T = len(test_source)
      print(T)
      extra_words = ((T // batch_size) + 1)*batch_size - T
      for i in range(extra_words) :
        temp = []
        for k in range(3) :
          temp.append(k+5)
        test_source.append(temp)
      
      assert len(test_source) == (extra_words+T)
      print(len(test_source)) 
  
      input_data = loaded_graph.get_tensor_by_name('input:0')
      logits = loaded_graph.get_tensor_by_name('predictions:0')
      target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
      keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
       
      pred_words = []
      
      for batch_i in range(0, len(test_source)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = test_source[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_vocab_to_int['<PAD>']))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source)) 
      
        output_logits = sess.run(logits, feed_dict = {input_data: pad_sources_batch,
                                         target_sequence_length: pad_source_lengths,
                                         keep_prob: 1.0})
        
        for i in range(len(output_logits)) :
          if i + start_i < T :
            cur_word = []
            for k in range(len(output_logits[i])) :
              if output_logits[i][k]==source_vocab_to_int['<EOW>'] :
                break
              else :
                cur_word.append(out_voc_int[output_logits[i][k]])
            pred_words.append(cur_word)
    
      for i in range(len(pred_words)) :
        str1=""
        pred_words[i] = str1.join(pred_words[i])
        pred_words[i] = pred_words[i].replace(" ","")
        pred_words[i] = " ".join(pred_words[i])
      words = np.array(pred_words)
    #index = (np.arange(0,X_test.shape[0]).T).reshape(X_test.shape[0],1)
      pd.DataFrame(words).to_csv("predictions.csv", header=["HIN"])

#predict(end_epoch)