import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')
train_data=train[['AMOUNT_MAX', 'PROBABILITY_ThirdQuartile', 'Repeater_Index_Max', 'Back_and_Forth_Counter_Max', 'Forward_Count_Max', 'Quarterly_Positive_Momentum_Max',
      'RATIO_REPEATER_TO_TOTAL', 'OVERALL_SLOPE', 'OVERALL_SLOPE_MIDPOINT', 'OPPORTUNITY_HUNG_FOR_LONG']]
train_target=train['TGT_LOST']
train_target1=pd.get_dummies(train_target).values
pca=PCA(n_components=10)
X=pca.fit_transform(train_data)

#figsize就是控制图片大小，可去掉
f=plt.figure(figsize=(60,60))
ax=f.add_subplot(111)
ax.scatter(X[:,0][train_target==0],X[:,1][train_target==0],c='r')
ax.scatter(X[:,0][train_target==1],X[:,1][train_target==1],c='y')
ax.set_title('数据分布图')
plt.show()

x=tf.placeholder(dtype=tf.float32,shape=[None,10],name="input")
y=tf.placeholder(dtype=tf.float32,shape=[None,2],name="output")

w=tf.get_variable("weight",shape=[10,2],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
bais=tf.get_variable("bais",shape=[2],dtype=tf.float32,initializer=tf.constant_initializer(0))
y_1=tf.nn.bias_add(tf.matmul(x,w),bais)

#labels是实际值actual value, logits是预测值pred value
loss=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_1))
x0min,x0max=X[:,0].min(),X[:,0].max()
x1min,x1max=X[:,1].min(),X[:,1].max()

with tf.Session() as sess:
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y,1),tf.arg_max(y_1,1)),tf.float32))
    train_step=tf.train.AdamOptimizer().minimize(loss)
    my=tf.arg_max( y_1,1)
    sess.run(tf.global_variables_initializer())
    for i in range(1000001):
        sess.run(train_step,feed_dict={x:X,y:train_target1})
        if i%500==0:
            accuracy_print=sess.run(accuracy,feed_dict={x:X,y:train_target1})
            print(accuracy_print)
    
    test_x=test[['AMOUNT_MAX', 'PROBABILITY_ThirdQuartile', 'Repeater_Index_Max', 'Back_and_Forth_Counter_Max', 'Forward_Count_Max', 'Quarterly_Positive_Momentum_Max',
      'RATIO_REPEATER_TO_TOTAL', 'OVERALL_SLOPE', 'OVERALL_SLOPE_MIDPOINT', 'OPPORTUNITY_HUNG_FOR_LONG']]
    pred=sess.run(my,feed_dict={x:test_x})

np.savetxt("predict.csv", pred);
#一共16707行test数据，预测准确的有13488行，准确率81%，训练时间3:30->5:30(2个小时),训练次数1000001

pred.shape
pred

f2=plt.figure(figsize=(60,60))
ax2=f2.add_subplot(111)
ax2.scatter(X[:,0][pred==0],X[:,1][pred==0],c='r')
ax2.scatter(X[:,0][pred==1],X[:,1][pred==1],c='y')
ax2.set_title('数据分布图')
plt.show()