from plot_signal import plot_signal
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

filename1 = "unlabelled/1131314/new9.txt"
#filename = "unlabelled/2987_2.asc"
#filename = "unlabelled/4264/waves4.asc"
filename2 = "unlabelled/2987_5.asc"

with open(filename1) as f:
    content1 = f.readlines()
with open(filename2) as f:
    content2 = f.readlines()

content2.pop(0)
content2.pop(0)
# you may also want to remove whitespace characters like `\n` at the end of each line
content1 = [float(x.replace('\\n', '').strip()) for x in content1]
content2 = [float(x.replace('\\n', '').strip()) for x in content2]
x, y, z = range(10000), np.array(np.concatenate((content1[2000:7000], content2[2000:7000]), axis=0)), np.ones(10000)
#y = np.array([n+np.mean(y) for n in y])
#y = np.array([(n--1)/(5--1) for n in y])
print(y)
n = np.lib.pad(y, (900, 900), 'edge')
temp = []
for i in range(len(n)-900):
    next_x = n[i:i+900]
    temp.append((next_x - np.min(next_x))/(np.max(next_x)-np.min(next_x)))
temp = np.array(temp)

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

for i in range(len(n)-1800):
    z[i] =  np.mean(output[i:i+900], axis=0)

plt.style.use('dark_background')
plot_signal(x, y, z)
plt.show()
