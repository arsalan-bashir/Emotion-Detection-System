***Emotion Detection and Recognition from Text Using Neural Networks***

The goal of this project is to create an algorithm that could classify the right emotion from text.

In this project I examined a few architectures of neural networks in two different approaches.
One approach is called “Bag of Words” and uses an **MLP model**.
The second approach is called “Sequence of Words” and uses a **CNN model**.

I used four different datasets to train the models, to compare the influence of each dataset on the results.
By using the “We Feel Fine Project” dataset, the model achieved high accuracy, in both MLP and CNN models, 
but the CNN model took much longer time to learn.

Additionally, in the other datasets, the CNN model couldn’t at all, and in the MLP model there were relatively good resultsin some of the datasets.
Datasets with a large amount of samples and a small number of classification options showed better results, 
but the number of samples in the dataset was the most influential for the success of the model.

**Datasets**

ISEAR
https://github.com/PoorvaRane/Emotion-Detector/blob/master/ISEAR.csv

IMDB
http://ai.stanford.edu/~amaas/data/sentiment/

Twitter
https://raw.githubusercontent.com/tlkh/text-emotion-classification/master/dataset/data/text_emotion.csv

We Feel Fine
http://www.wefeelfine.org/
