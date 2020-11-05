import gabor_face_recognition

base_url = "D:/backupData/JiangFanBackup_Tech/master/Find_Face_In_Video/python/Gabor_Face_Recognition/images"
# base_url = '../images'
img_url_02 = base_url + "/a/cherry02.jpg"
img_url_03 = base_url + "/b/cherry03.jpg"
# lena = mpimg.imread(img_url_02) # 读取和代码处于同一目录下的 lena.png
# lena.shape #(512, 512, 3)
# plt.imshow(lena) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()
v_res = gabor_face_recognition.get_hist(img_url_02, img_url_03)
print(v_res)
