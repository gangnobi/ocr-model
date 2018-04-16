import tensorflow as tf
import scipy.ndimage
from scipy.misc import imsave
import matplotlib.pyplot as plt
import numpy as np
import utils
import cv2

labels = utils.load_labels("model/output_labels.txt")
img = cv2.imread("test.jpg")
t = utils.read_tensor_from_opencv(255-img)

graph = utils.load_graph("model/output_graph.pb")

x = graph.get_operation_by_name(
    'import/module_apply_default/MobilenetV2/input')
y = graph.get_operation_by_name(
    'import/module_apply_default/MobilenetV2/expanded_conv_16/project/Conv2D')

persistent_sess = tf.Session(graph=graph)
y_out = persistent_sess.run(y.outputs[0], feed_dict={
    x.outputs[0]: t
})

print(y_out.shape)

size = 7*20, 7*16, 3 
m = np.zeros(size, dtype=np.float32)

for i in range(20):
    for j in range(16):
        size = 7, 7, 3
        gray = y_out[:, :, :, i*j].reshape(7, 7)
        img2 = np.zeros(size, dtype=np.float32)
        img2[:, :, 0] = gray
        img2[:, :, 1] = gray
        img2[:, :, 2] = gray
        m[7*i:7*(i+1), 7*j:7*(j+1), :] = img2

cv2.imshow("cov2.png", m)
cv2.waitKey(0)
