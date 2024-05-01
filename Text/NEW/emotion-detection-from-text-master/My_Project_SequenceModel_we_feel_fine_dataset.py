
# coding: utf-8

# # Imports

# In[1]:


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import collections
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D

import pickle
import argparse
import time
import itertools


# # Upload And Process Data

# In[2]:


# Upload we feel fine dataset:

my_data = pd.read_csv('project datasets\We_Feel_Fine_dataset.csv')

# shuffle data:
my_data = shuffle(my_data)

# display(my_data.sort_values('text'))
print("\ndata size: ", len(my_data))
print('display examples for each class:')
display(my_data.groupby('emotion').describe())

# drop duplicates:
my_data.drop_duplicates(subset='text', inplace=True)
my_data =  my_data.reset_index().iloc[:,1:3]

print("\ndata size after removing duplicates: ", len(my_data))
display(my_data.groupby('emotion').describe())


# In[3]:


# plot number of examples per class:

my_data.groupby('emotion').describe().plot.bar(y=1,rot=0, legend=False)
print("\ndata size: ", len(my_data))
plt.ylabel('number of samples')
plt.show()


# In[4]:


# remove examples of some classes to balance the data:

split_txt = my_data[my_data.emotion=='sadness'].text.str.split(' ')
my_data[my_data.emotion=='sadness']=my_data[my_data.emotion=='sadness'][split_txt.str.len()<14]

split_txt = my_data[my_data.emotion=='joy'].text.str.split(' ')
my_data[my_data.emotion=='joy']=my_data[my_data.emotion=='joy'][split_txt.str.len()<14]

split_txt = my_data[my_data.emotion=='anger'].text.str.split(' ')
my_data[my_data.emotion=='anger']=my_data[my_data.emotion=='anger'][split_txt.str.len()<30]

my_data = my_data[my_data.emotion !='surprise']

my_data = my_data.dropna()
display(my_data.groupby('emotion').describe())


# In[5]:


# display changes:

my_data.groupby('emotion').describe().plot.bar(y=1,rot=0, legend=False)
print("\ndata size: ", len(my_data))
plt.ylabel('number of samples')
plt.show()


# In[6]:


# removing blank examples:

len_texts = my_data['text'].str.split(' ').str.len()
# display(my_data[len_texts <=2 ])
split_txt = my_data.text.str.split(' ')
my_data=my_data[split_txt.str.len()> 2]
my_data.dropna(inplace=True)

# display(my_data[len_texts <= 3 ])
my_data = my_data[my_data.text != '[ For example']
my_data = my_data[my_data.text != '[ No description.]']
my_data = my_data[my_data.text != '[ No response.]']
my_data = my_data[my_data.text != '[ During inter-rail-trip']
my_data = my_data[my_data.text != '[ No reponse.]']
my_data = my_data[my_data.text != '[ No response.]']
my_data = my_data[my_data.text != '[ Never experienced.]']
my_data = my_data[my_data.text != '[ Never experienced.]']

my_data.dropna(inplace=True)
# display(my_data[len_texts <= 3 ])


# In[7]:


len_texts = my_data['text'].str.split(' ').str.len()

three_words = my_data.text[len_texts==3]

index_ = [i for i, n in enumerate(three_words) if 'When' in n]
for j in sorted(index_, reverse=True):
  my_data.drop(three_words.index[j], inplace=True)
  
my_data =  my_data.reset_index().iloc[:,1:3]


# In[8]:


# Upload and process Tweeter Dataset:
df1 = pd.read_csv('https://query.data.world/s/b6rddgtaxc7mqfzalplotof7ft7qst')
df1 = df1[['sentiment','content']]
df1.rename(index=str, columns={"sentiment": "emotion", "content": "text"}, inplace=True)

# drop duplicates:
df1.drop_duplicates(subset='text', inplace=True)
df1 =  df1.reset_index().iloc[:,1:3]

# take out only the 'love' class:
df1_love = df1[df1.emotion=='love']
df1_love =  df1_love.reset_index().iloc[:,1:3]


# In[9]:


# remove some random chars to avoid noise and hidden duplicates:

for txt in range(len(df1_love['text'])):
  sentence = df1_love['text'].iloc[txt]
  sentence_splt=sentence.split()

  if '#' in sentence:
    index = [i for i, n in enumerate(sentence_splt) if '#' in n]
    for j in sorted(index, reverse=True): 
      del sentence_splt[j]

  if '$' in sentence:
    index = [i for i, n in enumerate(sentence_splt) if '$' in n]
    for j in sorted(index, reverse=True): 
      del sentence_splt[j]

  if '&' in sentence:
    index = [i for i, n in enumerate(sentence_splt) if '&' in n]
    for j in sorted(index, reverse=True): 
      del sentence_splt[j]

  if '@' in sentence:
    index = [i for i, n in enumerate(sentence_splt) if '@' in n]
    for j in sorted(index, reverse=True): 
      del sentence_splt[j]

  if 'ï¿½' in sentence:
    index = [i for i, n in enumerate(sentence_splt) if 'ï¿½' in n] 
    for j in sorted(index, reverse=True): 
      del sentence_splt[j]

  if 'http' in sentence:
    index = [i for i, n in enumerate(sentence_splt) if 'http' in n] 
    for j in sorted(index, reverse=True): 
      del sentence_splt[j]

  index_ = [i for i, n in enumerate(sentence_splt) if n == ''] 
  for j in sorted(index_, reverse=True): 
    del sentence_splt[j]


  if len(sentence_splt) == 0:
    sentence = np.nan
  else:
    df1_love['text'].iloc[txt] = ' '.join(sentence_splt)


# In[10]:


# concatenate to one dataset:
my_data = pd.concat([my_data, df1_love], ignore_index=True)

# shuffle data:
my_data = shuffle(my_data)

# drop duplicates:
my_data.drop_duplicates(subset='text', inplace=True)
my_data =  my_data.reset_index().iloc[:,1:3]


# # Display Data

# In[11]:


print("\ndata size: ", len(my_data))
print('number of classes: ', len(my_data.groupby('emotion').describe()))
print('average number of samples per class: %.2f' %(my_data.groupby('emotion').describe().iloc[:,0].mean()))
display(my_data.groupby('emotion').describe().iloc[:,[0,2]])

my_data.groupby('emotion').describe().plot.bar(y=1,rot=0, legend=False)
plt.ylabel('number of samples')
plt.show()


# In[12]:


# checking the length of the texts in the dataset:

len_texts = my_data['text'].str.split(' ').str.len()
print('median text length:', np.median(len_texts))
print('mean text length:', np.mean(len_texts))
print('min text length:', np.min(len_texts))
print('max text length:', np.max(len_texts))


# In[13]:


ratio = len(my_data)/np.mean(len_texts)
print('(Data Size)/(Average Text Length) Ratio: ' ,ratio)


# In[14]:


# creating a dictionary of the classes and converting classes into numbers:

emotions = np.unique(my_data['emotion'])
emotions_dict = dict((emotions[i], i) for i in range(len(emotions)))
print(emotions_dict)
my_data.replace(emotions,list(range(len(emotions))), inplace=True)
print('data size: ',len(my_data)) 
display(my_data.head())


# In[15]:


# plotting text length distribution:

len_texts = my_data['text'].str.split(' ').str.len()
plt.hist(len_texts, 50, color = 'c')  # num of words
plt.xlabel('Length of text')
plt.ylabel('Number of texts')
plt.title('Text length distribution')
plt.show()


# In[16]:


# plotting the frequency of words:

def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50):
    """Plots the frequency distribution of n-grams.
    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.figure(figsize=(30,5))
    plt.bar(idx, counts, width=0.8, color='c')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()
    
    return ngrams, counts
    
ngrams, counts = plot_frequency_distribution_of_ngrams(my_data['text'])


# # Divide And Tokenize Data

# In[17]:


# Taking 5% of the data for test set:

tr = int(np.ceil(len(my_data)*0.95))
ts = int(np.floor(len(my_data)*0.05))

train_data = my_data.head(tr)
print('training data size: ', len(train_data))
display(train_data.groupby('emotion').describe())

test_data = my_data.tail(ts)
test_data.reset_index(drop=True, inplace = True)
print('\n\ntest data size: ', len(test_data))
display(test_data.groupby('emotion').describe())


# In[18]:


# Vectorization parameters
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K,  filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ 1234567890')
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index, tokenizer, max_length


# In[19]:


# Split dataset to 95% training and 5% validation sets:

train_texts, val_texts, train_labels, val_labels = train_test_split(train_data['text'], train_data['emotion'], test_size=0.05, random_state=42)

# Verify that validation labels are in the same range as training labels.
num_classes = len(np.unique(train_labels))
unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
if len(unexpected_labels):
    raise ValueError('Unexpected label values found in the validation set:'
                     ' {unexpected_labels}. Please make sure that the '
                     'labels in the validation set are in the same range '
                     'as training labels.'.format(
                         unexpected_labels=unexpected_labels))

# Vectorize texts.
x_train, x_val, word_index, tokenizer, max_length = sequence_vectorize(
    train_texts, val_texts)

# Number of features will be the embedding input dimension. Add 1 for the
# reserved index 0.
num_features = min(len(word_index) + 1, TOP_K)

x_test = tokenizer.texts_to_sequences(test_data['text'])
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

y_test = list(test_data['emotion'])


# In[20]:


# Saving the objects:
with open('saved models\sequence_wordindex_N_tokenizer.pkl', 'wb') as f: 
    pickle.dump([word_index, tokenizer, max_length], f)


# In[21]:


print('Train examples: ', x_train.shape[0])
print('Validation examples: ', x_val.shape[0])
print('Test examples: ', x_test.shape[0])


# In[22]:


np.shape(x_train)


# # Build And Train Model

# In[23]:


# Depthwise Separable CNN Model:

def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features,
                 embedding_matrix=None):
    """Creates an instance of a separable CNN model.

    # Arguments
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of the layers.
        kernel_size: int, length of the convolution window.
        embedding_dim: int, dimension of the embedding vectors.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.
        num_features: int, number of words (embedding input dimension).
        embedding_matrix: dict, dictionary with embedding coefficients.

    # Returns
        A sepCNN model instance.
    """
    model = models.Sequential()

    # Add embedding layer.
    model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0]))

    for _ in range(blocks-1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model


# In[24]:


# Training function:

def train_sequence_model(x_train,
                         train_labels,
                         x_val,
                         val_labels,
                         num_features,
                         learning_rate=1e-3,
                         epochs=1000,
                         batch_size=128,
                         blocks=2,
                         filters=64,
                         dropout_rate=0.2,
                         embedding_dim=200,
                         kernel_size=3,
                         pool_size=3):
    """Trains sequence model on the given dataset.

    # Arguments
        x_train: vectorized texts of training texts
        train_labels: list of ints of training labels
        x_val: vectorized texts of validation texts
        val_labels: list of ints of validation labels
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.

    # Returns
        Trained sepCNN model, history object.
    """

    # Create model instance.
    model = sepcnn_model(blocks=blocks,
                       filters=filters,
                       kernel_size=kernel_size,
                       embedding_dim=embedding_dim,
                       dropout_rate=dropout_rate,
                       pool_size=pool_size,
                       input_shape=x_train.shape[1:],
                       num_classes=num_classes,
                       num_features=num_features)


    loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Save model.
    model.save('saved models\we_feel_fine_sepcnn_model.h5')
    return model, history


# In[25]:


# Calling Training Function:

sequential_sepCNN_model, history = train_sequence_model(x_train,
                                                       train_labels,
                                                       x_val,
                                                       val_labels,
                                                       num_features,
                                                       learning_rate=0.001,
                                                       epochs=100,
                                                       batch_size=200,
                                                       blocks=2,
                                                       filters=64,
                                                       dropout_rate=0.5,
                                                       embedding_dim=200,
                                                       kernel_size=3,
                                                       pool_size=3)


# # Display Results

# In[26]:


# Print results.
history = history.history
print('\nTraining Accuracy: {}\nValidation Accuracy: {}'.format(
        history['acc'][-1], history['val_acc'][-1]))

# summarize history for accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['train acc', 'validation acc'], loc='upper left')
plt.show()

print('\nTraining Loss: {}\nValidation Loss: {}'.format(
        history['loss'][-1], history['val_loss'][-1]))
# Output a graph of loss metrics over periods.
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Model Loss")
plt.tight_layout()
plt.plot(history['loss'], label="training loss")
plt.plot(history['val_loss'], label="validation loss")
plt.legend()
plt.show()


# In[27]:


# Plot Confusion Matrix:
    
test_prob = sequential_sepCNN_model.predict(x_test).tolist()
test_pred = np.zeros((len(test_prob),))
for y in range(len(test_prob)):
  test_pred[y] = test_prob[y].index(np.max(test_prob, axis=1)[y])
 
cm_num = metrics.confusion_matrix(y_test, test_pred)
# print('confusion matrix:\n', cm_num)

# recall = how many examples the model was right to predict as positive, out of *all the positive examples* in that class.
# precision = how many examples the model was right to predict as positive, out of all the *examples it predicted as positive* in that class.
precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, test_pred)
precision = np.expand_dims(precision, axis=0)
recall = np.append(recall,[0])
recall = np.expand_dims(recall, axis=1)


# Normalize the confusion matrix by row (i.e by the number of samples in each class):
cm = cm_num.astype("float") / cm_num.sum(axis=1)[:, np.newaxis]
cm = np.concatenate((cm, precision), axis=0)
cm = np.append(cm, recall, axis=1)

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmax=1)
plt.colorbar()
fmt = '.2f'
thresh = 0.5
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.title("Normalized Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.xticks(list(range(len(emotions)+1)), np.append(emotions,['RECALL']), rotation=45)
plt.yticks(list(range(len(emotions)+1)), np.append(emotions,['PRECISION']))
plt.grid(False)
plt.show()


# In[28]:


loss, accuracy = sequential_sepCNN_model.evaluate(x_test, y_test)
print('test loss : %.2f\ntest accuracy: %.2f ' %(loss, accuracy))


# # Random Guess

# In[29]:


print('probability of examples of every emotion in the dataset:\n')

len_emotions=[]
for i in range(len(emotions)):
  len_emotions.append(len(my_data[my_data['emotion']==i]))
  print('%s: %.2f' %(emotions[i], len_emotions[i]/len(my_data)))


# In[30]:


# plot confusion matrix of random guess for comparison

rand_guess = y_test.copy()
np.random.shuffle(rand_guess)

cm_num = metrics.confusion_matrix(y_test, rand_guess)
print('confution matrix:\n', cm_num)

# recall = how many examples the model was right to predict as positive, out of *all the positive examples* in that class.
# precision = how many examples the model was right to predict as positive, out of all the *examples it predicted as positive* in that class.
precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, rand_guess)
precision = np.expand_dims(precision, axis=0)
recall = np.append(recall,[0])
recall = np.expand_dims(recall, axis=1)


# Normalize the confusion matrix by row (i.e by the number of samples in each class):
cm = cm_num.astype("float") / cm_num.sum(axis=1)[:, np.newaxis]
cm = np.concatenate((cm, precision), axis=0)
cm = np.append(cm, recall, axis=1)

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmax=1)
plt.colorbar()
fmt = '.2f'
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > 0.5 else "black")
plt.title("Normalized Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.xticks(list(range(len(emotions)+1)), np.append(emotions,['recall']))
plt.yticks(list(range(len(emotions)+1)), np.append(emotions,['precision']))
plt.grid(False)
plt.show()


# In[31]:


loss, accuracy = sequential_sepCNN_model.evaluate(x_test, rand_guess)
print('test loss : %.2f\ntest accuracy: %.2f ' %(loss, accuracy))


# # Summary

# In[33]:


display(sequential_sepCNN_model.summary())

