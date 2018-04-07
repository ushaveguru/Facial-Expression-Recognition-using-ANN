from __future__ import print_function, division
from builtins import range


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import getData, getBinaryData, y2indicator, error_rate, init_weight_and_bias
from sklearn.utils import shuffle

n_nodes_h1=2000
n_nodes_h2=1000
n_nodes_h3=500
learning_rate=10e-7
mu=0.99
decay=0.999
reg=10e-3
epochs=400
batch_sz=100
params = []

X, Y = getData()

def main():
    fit(X, Y, show_fig=True)




def fit(X, Y, show_fig=False):
    
    K = len(set(Y)) 
    # make a validation set
    X, Y = shuffle(X, Y)
    X = X.astype(np.float32)
    Y = y2indicator(Y).astype(np.float32)
    Xvalid, Yvalid = X[-1000:], Y[-1000:]
    Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate
    X, Y = X[:-1000], Y[:-1000]

    N, D = X.shape
    tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
    tfT = tf.placeholder(tf.float32, shape=(None, K), name='T')
    prediction = neural_network(D,K,tfX)


    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction,
                labels=tfT
            )
        ) 
    
    
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)

    

    n_batches = N // batch_sz
    epoch_loss = 0
    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})
                if j % 20 == 0:
                    c = session.run([train_op,cost],feed_dict={tfX: Xbatch, tfT: Ybatch})
                    costs.append(c)
                    p = session.run(tf.argmax(prediction,1), feed_dict={tfX: Xvalid, tfT: Yvalid})
                    e = error_rate(Yvalid_flat, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

    

        if show_fig:
            plt.plot(costs)
            plt.show()


def neural_network(D,K,tfX):
    W1, b1 = init_weight_and_bias(D, n_nodes_h1)
    hidden_1_layer={'w': tf.Variable(W1.astype(np.float32)), 'b':tf.Variable(b1.astype(np.float32))}
    W2, b2 = init_weight_and_bias(n_nodes_h1, n_nodes_h2)
    hidden_2_layer={'w': tf.Variable(W2.astype(np.float32)), 'b':tf.Variable(b2.astype(np.float32))}
    W3, b3 = init_weight_and_bias(n_nodes_h2, n_nodes_h3)
    hidden_3_layer={'w': tf.Variable(W3.astype(np.float32)), 'b':tf.Variable(b3.astype(np.float32))}
    W4, b4 = init_weight_and_bias(n_nodes_h3, K)
    output_layer={'w': tf.Variable(W4.astype(np.float32)), 'b':tf.Variable(b4.astype(np.float32))}

#forwarding
    l1 = tf.add(tf.matmul(tfX, hidden_1_layer['w']) , hidden_1_layer['b'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['w']) , hidden_2_layer['b'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['w']) , hidden_3_layer['b'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['w'] + output_layer['b'])

    params.extend([W1,b1,W2,b2,W3,b3,W4,b4])

    return output





if __name__ == '__main__':
    main()
