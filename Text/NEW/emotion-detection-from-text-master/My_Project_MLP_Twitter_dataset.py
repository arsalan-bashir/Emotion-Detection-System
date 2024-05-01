
# coding: utf-8

# # Imports

# In[1]:


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import wordcloud
import itertools
import pickle


# # Upload And Process Data

# In[3]:


# Twitter Dataset:
Twitter_data = pd.read_csv('project datasets\Twitter_dataset.csv')

# select relevant columns and replace names:
Twitter_data = Twitter_data[['sentiment','content']]
Twitter_data.rename(index=str, columns={"sentiment": "emotion", "content": "text"}, inplace=True)

# plot numbet of examples per class:
Twitter_data.groupby('emotion').describe().plot.bar(y=1,rot=45, legend=False)
plt.ylabel('number of samples')
plt.show()

# shuffle and display data:
Twitter_data = shuffle(Twitter_data)
print("\ndata size: ", len(Twitter_data))
display(Twitter_data.groupby('emotion').describe().iloc[:,0:3])

# drop duplicates:
Twitter_data.drop_duplicates(subset='text', inplace=True)
Twitter_data =  Twitter_data.reset_index().iloc[:,1:3]

# remove some classes:
Twitter_data = Twitter_data[Twitter_data.emotion !='empty']
Twitter_data = Twitter_data[Twitter_data.emotion !='neutral']
Twitter_data = Twitter_data[Twitter_data.emotion !='relief']
Twitter_data = Twitter_data[Twitter_data.emotion !='surprise']
Twitter_data = Twitter_data[Twitter_data.emotion !='enthusiasm']
Twitter_data = Twitter_data[Twitter_data.emotion !='fun']
Twitter_data = Twitter_data[Twitter_data.emotion !='boredom']

# join classes:
Twitter_data = Twitter_data.replace('happiness', 'joy')
Twitter_data = Twitter_data.replace('hate', 'anger')

# display data after changes:
print("\ndata size: ", len(Twitter_data))
Twitter_data.groupby('emotion').describe().plot.bar(y=1,rot=45, legend=False)
plt.ylabel('number of samples')
plt.show()
display(Twitter_data.groupby('emotion').describe().iloc[:,0:3])


# In[4]:


# remove examples of some classes to balance the data:

split_txt = Twitter_data[Twitter_data.emotion=='sadness'].text.str.split(' ')
Twitter_data[Twitter_data.emotion=='sadness']=Twitter_data[Twitter_data.emotion=='sadness'][split_txt.str.len()<15]

split_txt = Twitter_data[Twitter_data.emotion=='joy'].text.str.split(' ')
Twitter_data[Twitter_data.emotion=='joy']=Twitter_data[Twitter_data.emotion=='joy'][split_txt.str.len()<14]

split_txt = Twitter_data[Twitter_data.emotion=='worry'].text.str.split(' ')
Twitter_data[Twitter_data.emotion=='worry']=Twitter_data[Twitter_data.emotion=='worry'][split_txt.str.len()<13]

Twitter_data = Twitter_data.dropna()


# In[5]:


# display changes:
Twitter_data.groupby('emotion').describe().plot.bar(y=1,rot=45, legend=False)
plt.ylabel('number of samples')
plt.show()


# In[6]:


# upload and process ISEAR data:

ISEAR_data = pd.read_csv('https://raw.githubusercontent.com/PoorvaRane/Emotion-Detector/master/ISEAR.csv', delimiter=',', header=None, names=['emotion','text','none'])
ISEAR_data = ISEAR_data.iloc[:, 0:2]
print('data size: ',len(ISEAR_data))

print(np.unique(ISEAR_data['emotion']))
ISEAR_data = ISEAR_data.replace('guit', 'guilt')
print(np.unique(ISEAR_data['emotion']))

ISEAR_data.drop_duplicates(subset='text', inplace=True)
display(ISEAR_data.groupby('emotion').describe())
print('\ndata size: ',len(ISEAR_data))


# In[7]:


# join and remove unneeded classes of ISEAR dataset:
ISEAR_data = ISEAR_data[ISEAR_data.emotion !='guilt']
ISEAR_data = ISEAR_data[ISEAR_data.emotion !='fear']
ISEAR_data = ISEAR_data[ISEAR_data.emotion !='shame']

ISEAR_data = ISEAR_data.replace('disgust', 'anger')
display(ISEAR_data.groupby('emotion').describe())


# In[8]:


# concatenate to one dataset:
my_data = pd.concat([Twitter_data, ISEAR_data], ignore_index=True)

# shuffle data:
my_data = shuffle(my_data)

# drop duplicates:
my_data.drop_duplicates(subset='text', inplace=True)
my_data =  my_data.reset_index().iloc[:,1:3]

# display data:
print("\ndata size: ", len(my_data))
display(my_data.groupby('emotion').describe())


# In[9]:


# remove some random chars to avoid noise and hidden duplicates:

for txt in range(len(my_data['text'])):
  text = my_data['text'].iloc[txt]
  text_splt=text.split()

  if '#' in text:
    index = [i for i, n in enumerate(text_splt) if '#' in n]
    for j in sorted(index, reverse=True): 
      del text_splt[j]

  if '$' in text:
    index = [i for i, n in enumerate(text_splt) if '$' in n]
    for j in sorted(index, reverse=True): 
      del text_splt[j]

  if '&' in text:
    index = [i for i, n in enumerate(text_splt) if '&' in n]
    for j in sorted(index, reverse=True): 
      del text_splt[j]

  if '@' in text:
    index = [i for i, n in enumerate(text_splt) if '@' in n]
    for j in sorted(index, reverse=True): 
      del text_splt[j]

  if 'ï¿½' in text:
    index = [i for i, n in enumerate(text_splt) if 'ï¿½' in n] 
    for j in sorted(index, reverse=True): 
      del text_splt[j]

  if 'http' in text:
    index = [i for i, n in enumerate(text_splt) if 'http' in n] 
    for j in sorted(index, reverse=True): 
      del text_splt[j]

  index_ = [i for i, n in enumerate(text_splt) if n == ''] 
  for j in sorted(index_, reverse=True): 
    del text_splt[j]

  my_data['text'].iloc[txt] = ' '.join(text_splt)


# In[10]:



print("\ndata size: ", len(my_data))
display(my_data.groupby('emotion').describe())

# drop duplicates:
my_data.drop_duplicates(subset='text', inplace=True)
my_data =  my_data.reset_index().iloc[:,1:3]

display(my_data.groupby('emotion').describe())
print("\ndata size after cleaning duplicates: ", len(my_data))


# In[11]:


# removing blank examples:

len_texts = my_data['text'].str.split(' ').str.len()

# display(my_data[len_texts < 4 ])
my_data = my_data[my_data.text != 'None.']
my_data = my_data[my_data.text != 'Blank.']
my_data = my_data[my_data.text != '[ No description.]']
my_data = my_data[my_data.text != '[ No response.]']
my_data = my_data[my_data.text != '[ No reponse.]']
my_data = my_data[my_data.text != 'Nothing.']
my_data = my_data[my_data.text != 'Not applicable.']
my_data = my_data[my_data.text != '[ Not applicable.]']
my_data = my_data[my_data.text != 'NO RESPONSE.']
my_data = my_data[my_data.text != 'NO RESPONSE']
my_data = my_data[my_data.text != 'Earth crake.']
my_data = my_data[my_data.text != "Doesn't apply."]
my_data = my_data[my_data.text != 'Does not apply.']
my_data = my_data[my_data.text != 'DO NOT REMEMBER.']
my_data = my_data[my_data.text != 'During a meeting.']
my_data = my_data[my_data.text != 'Before an examn.']
my_data = my_data[my_data.text != 'As for disgust.']
my_data = my_data[my_data.text != 'At set-backs.']
my_data = my_data[my_data.text != '[ Do not know.]']

my_data.dropna(inplace=True)
# display(my_data[len_texts < 4 ])


# # Display Data

# In[12]:


print("\ndata size: ", len(my_data))
print('number of classes: ', len(my_data.groupby('emotion').describe()))
print('average number of samples per class: %.2f' %(my_data.groupby('emotion').describe().iloc[:,0].mean()))
display(my_data.groupby('emotion').describe().iloc[:,[0,2]])

print("\ndata size: ", len(my_data))
my_data.groupby('emotion').describe().plot.bar(y=1,rot=0, legend=False)
plt.ylabel('number of samples')
plt.show()


# In[13]:


# checking the length of the texts in the dataset:

len_texts = my_data['text'].str.split(' ').str.len()
print('median text length:', np.median(len_texts))
print('mean text length:', np.mean(len_texts))
print('min text length:', np.min(len_texts))
print('max text length:', np.max(len_texts))


# In[14]:


ratio = len(my_data)/np.mean(len_texts)
print('(Data Size)/(Average Text Length) Ratio: ' ,ratio)


# In[15]:


# creating a dictionary of the classes and converting classes into numbers:

emotions = np.unique(my_data['emotion'])
emotions_dict = dict((emotions[i], i) for i in range(len(emotions)))
print(emotions_dict)
my_data.replace(emotions,list(range(len(emotions))), inplace=True)
display(my_data.head())


# In[16]:


# plotting text length distribution:

plt.hist(len_texts, 50, color = 'c')  # num of words
plt.xlabel('Length of text')
plt.ylabel('Number of texts')
plt.title('Text length distribution')
plt.show()


# In[17]:


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

# In[18]:


# Taking 15% of the data for test set:

tr = int(np.ceil(len(my_data)*0.85))
ts = int(np.floor(len(my_data)*0.15))

train_data = my_data.head(tr)
print('training data size: ', len(train_data))
display(train_data.groupby('emotion').describe())

test_data = my_data.tail(ts)
test_data.reset_index(drop=True, inplace = True)
print('\n\ntest data size: ', len(test_data))
display(test_data.groupby('emotion').describe())


# In[19]:


# Vectorization parameters

# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 10000


# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 3

def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as ngram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',          # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)
    
    x = x_train.copy()
    f, p = f_classif(x, train_labels)
    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)


    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    return x_train, x_val, vectorizer, selector, f, p, x


# In[20]:


# Split dataset to 80% training and 20% validation sets:

train_texts, val_texts, train_labels, val_labels = train_test_split(train_data['text'], train_data['emotion'], test_size=0.2, random_state=42)

# Verify that validation labels are in the same range as training labels.
num_classes = len(np.unique(train_labels))
unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
if len(unexpected_labels):
    raise ValueError('Unexpected label values found in the validation set:'
                     ' {unexpected_labels}. Please make sure that the '
                     'labels in the validation set are in the same range '
                     'as training labels.'.format(
                         unexpected_labels=unexpected_labels))

# vectorize texts for training, validation and test sets:
x_train, x_val, vectorizer, selector, f, p, x = ngram_vectorize(
        train_texts, train_labels, val_texts)

x_test = vectorizer.transform(test_data['text'])
x_test = selector.transform(x_test)
x_test = x_test.astype('float32')

y_test = list(test_data['emotion'])


# In[21]:


print('Train examples: ', x_train.shape[0])
print('Validation examples: ', x_val.shape[0])
print('Test examples: ', x_test.shape[0])


# In[22]:


print('vocabulary size: ' ,len(vectorizer.get_feature_names()))


# In[23]:


display(x_train)


# # Build And Train Model

# In[24]:


# MultiLayer Perceptron Model:
def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_classes, activation='softmax'))
    return model


# In[25]:


# Training function:

def train_ngram_model(x_train,
                      train_labels,
                      x_val,
                      val_labels,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains MLP Bag Of Words model on the given dataset.

    # Arguments
        x_train: vectorized texts of training texts
        train_labels: list of ints of training labels
        x_val: vectorized texts of validation texts
        val_labels: list of ints of validation labels
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
    # Returns
        Trained MLP model.
    """

    # Create model instance.
    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes)

    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

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
    model.save('saved models\Twitter_MLP_Model.h5')
    return model, history


# In[26]:


# Calling Training Function:

emotion_classification_model, history = train_ngram_model(x_train,
                                                          train_labels,
                                                          x_val,
                                                          val_labels,
                                                          learning_rate=0.001,
                                                          epochs=200,
                                                          batch_size=128,
                                                          layers=2,
                                                          units=64,
                                                          dropout_rate=0.5)


# # Display Results

# In[27]:


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


# In[29]:


# Plot Confution Matrix:

test_prob = emotion_classification_model.predict(x_test).tolist()
test_pred = np.zeros((len(test_prob),))
for y in range(len(test_prob)):
 test_pred[y] = test_prob[y].index(np.max(test_prob, axis=1)[y])

cm_num = metrics.confusion_matrix(y_test, test_pred)
print('confution matrix:\n', cm_num)

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


# In[30]:


loss, accuracy = emotion_classification_model.evaluate(x_test, y_test)
print('test loss : %.2f\ntest accuracy: %.2f ' %(loss, accuracy))


# # Rand Guess

# In[31]:


print('probability of examples of every emotion in the dataset:\n')

len_emotions=[]
for i in range(len(emotions)):
  len_emotions.append(len(my_data[my_data['emotion']==i]))
  print('%s: %.2f' %(emotions[i], len_emotions[i]/len(my_data)))


# In[32]:


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


# In[33]:


loss, accuracy = emotion_classification_model.evaluate(x_test, rand_guess)
print('rand guess loss : %.2f\nrand guess accuracy: %.2f ' %(loss, accuracy))


# # Summary

# In[34]:


display(emotion_classification_model.summary())

