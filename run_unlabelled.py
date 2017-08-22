from plot_signal import plot_signal
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

filename1 = "sah/3022_4.asc"
#filename = "unlabelled/2987_2.asc"
#filename = "unlabelled/4264/waves4.asc"
filename2 = "sah/2987_2.asc"

with open(filename1) as f:
    content1 = f.readlines()
with open(filename2) as f:
    content2 = f.readlines()

content1.pop(0)
content1.pop(0)
content2.pop(0)
content2.pop(0)
# you may also want to remove whitespace characters like `\n` at the end of each line
content1 = [float(x.replace('\\n', '').strip()) for x in content1]
content2 = [float(x.replace('\\n', '').strip()) for x in content2]
x, y, z = range(1000), np.array(np.concatenate((content1[200:700], content2[200:700]), axis=0)), np.ones(1000)
#y = np.array([n+np.mean(y) for n in y])
#y = np.array([(n--1)/(5--1) for n in y])
print(y)
n = np.lib.pad(y, (90, 90), 'edge')
temp = []
for i in range(len(n)-90):
    next_x = n[i:i+90]
    temp.append((next_x - np.min(next_x))/(np.max(next_x)-np.min(next_x)))
temp = np.array(temp)
'''
sess = tf.Session()

saver = tf.train.import_meta_graph('model/ann.meta')
saver.restore(sess, tf.train.latest_checkpoint('model/'))
# Assign the graph to all variables
graph = tf.get_default_graph()
x_place = graph.get_tensor_by_name("x:0")
run = graph.get_tensor_by_name("run:0")


output = sess.run(run, feed_dict={x_place:temp})
output = [n[0] - n[1] for n in output]
output = [(n--0.3)/(0.3--0.3) for n in output]
z = np.zeros(len(y))
output = np.array(output)

for i in range(len(n)-180):
    z[i] =  np.mean(output[i:i+90], axis=0)
'''
plt.style.use('dark_background')
plot_signal(x, y, z)
plt.show()
