import gabor_face_recognition
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

img_url = '../images/cherry02.png';
lena = mpimg.imread(img_url) # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)

plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

v_res = gabor_face_recognition.img2vector(img_url)
print(v_res)
