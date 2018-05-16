# 原始链接：https://gist.githubusercontent.com/sjchoi86/1757dd2fadba31393ab65c9f43af19ab/raw/9b429a96dd80540637dffb4f9b303c1caa484acb/mlp.py
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
%matplotlib inline  
print ("PACKAGES LOADED")

train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')
train_data=train[['AMOUNT_MAX', 'PROBABILITY_ThirdQuartile', 'Repeater_Index_Max', 'Back_and_Forth_Counter_Max', 'Forward_Count_Max', 'Quarterly_Positive_Momentum_Max',
      'RATIO_REPEATER_TO_TOTAL', 'OVERALL_SLOPE', 'OVERALL_SLOPE_MIDPOINT', 'OPPORTUNITY_HUNG_FOR_LONG']]
train_target=train['TGT_LOST']
train_target1=pd.get_dummies(train_target).values
test_x=test[['AMOUNT_MAX', 'PROBABILITY_ThirdQuartile', 'Repeater_Index_Max', 'Back_and_Forth_Counter_Max', 'Forward_Count_Max', 'Quarterly_Positive_Momentum_Max',
      'RATIO_REPEATER_TO_TOTAL', 'OVERALL_SLOPE', 'OVERALL_SLOPE_MIDPOINT', 'OPPORTUNITY_HUNG_FOR_LONG']]
test_target=test['TGT_LOST']

# NETWORK TOPOLOGIES
n_hidden_1 = 256 
n_hidden_2 = 128 
n_input    = 10 
n_classes  = 2  

# INPUTS AND OUTPUTS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
    
# NETWORK PARAMETERS
stddev = 0.1
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print ("NETWORK READY")

def mlp(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    return (tf.matmul(layer_2, _weights['out']) + _biases['out'])
  
# PREDICTION
pred = mlp(x, weights, biases)

# LOSS AND OPTIMIZER
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) 
# optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost) 
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) 
result_my=tf.arg_max(pred, 1)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(corr, "float"))

# INITIALIZER
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")

# PARAMETERS
training_epochs = 20
batch_size      = 100
display_step    = 4
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)


# OPTIMIZE
# 一共16707行test数据，预测准确的有13599行，准确率82%，训练时间1分钟，训练次数10
with tf.Session() as sess:
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y,1),tf.arg_max(pred,1)),tf.float32))
    training_step=tf.train.AdamOptimizer().minimize(cost)
    result_my=tf.arg_max(pred,1)
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(training_step,feed_dict={x:train_data,y:train_target1})
        if i%1000==0:
            accuracyPringing=sess.run(accuracy,feed_dict={x:train_data,y:train_target1})
            print(accuracyPringing)
    
    pred=sess.run(result_my,feed_dict={x:test_x})
    np.savetxt("predict.csv", pred);

# SAVE
#trainimg   = mnist.train.images
#trainlabel = mnist.train.labels
#testimg    = mnist.test.images
#testlabel  = mnist.test.labels
#w1   = sess.run(weights['h1'])
#w2   = sess.run(weights['h2'])
#wout = sess.run(weights['out'])
#b1   = sess.run(biases['b1'])
#b2   = sess.run(biases['b2'])
#bout = sess.run(biases['out'])

# SAVE TO MAT FILE 
#savepath = './data/mlp.mat'
#scipy.io.savemat(savepath
#   , mdict={'trainimg': trainimg, 'trainlabel': trainlabel
#           , 'testimg': testimg, 'testlabel': testlabel
#           , 'w1': w1, 'w2': w2, 'wout': wout, 'b1': b1, 'b2': b2, 'bout': bout})
#print ("%s SAVED." % (savepath))
