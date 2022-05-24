import numpy as np
import matplotlib.pyplot as plt


# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(5, 3.5), dpi=200)
plt.grid(axis='y')
plt.ylim((0.16, 0.30))
plt.xticks([])
#plt.ylim((0.1, 0.16))
# 再创建一个规格为 1 x 1 的子图
# plt.subplot(1, 1, 1)
# 柱子总数
N = 3
# 包含每个柱子对应值的序列
#values = (0.1548, 0.1476, 0.1325, 0.1316)
#values = (0.274, 0.256, 0.228)
values=(0.274, 0.224, 0.198)
#values=(0.1548, 0.1377, 0.1122)
#values = (0.1548, 0.1447, 0.1336, 0.1199)
#values = (0.1548,0.146,0.135,0.1143)
#values = (0.1647, 0.134, 0.139, 0.115)
# 包含每个柱子下标的序列
index = np.arange(N)
# 柱子的宽度
width = 0.5
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
plt.bar(index[0], values[0], width, color="#808000",label="Complete model",zorder=100)
plt.bar(index[1], values[1], width, color="#99CCFF",label="Only St",zorder=100)
plt.bar(index[2], values[2], width, color="#C8C8C8",label="Only Sst",zorder=100)
#plt.bar(index[3], values[3], width, color="#7bc8f6",label="w/o Dislike",zorder=100)
#plt.bar(index[3], values[3], width, color="#cfaf7b",label="w/o Dislike",zorder=100)
#plt.bar(index[4], values[4], width, color="#7bc8f6",label="only target",zorder=100)
# 设置横轴标签
# 设置纵轴标签
plt.ylabel('recall@80',fontsize=20)
# 添加纵横轴的刻度
plt.yticks(fontsize=17)
# 添加图例
plt.legend(fontsize=16)
#plt.savefig('/Users/gushuyun/Desktop/xiaorong1_Beibei.jpg')
plt.show()










"""
import matplotlib.pyplot as plt

#折线图
x = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]#点的横坐标
k1 = [0.218, 0.227, 0.2743, 0.255, 0.226, 0.166]#线1的纵坐标
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.ylim((0, 0.3))
plt.plot(x,k1,'o-',color = 'r',label="ATT-RLSTM")#s-:方形
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("lamda")#横坐标名字
plt.ylabel("racall@80")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
"""


"""
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.figure(figsize=(5, 3.5), dpi=200)
plt.grid(axis='x', ls='--')
plt.grid(axis='y', ls='--')
plt.ylim((0.1, 0.3))
x = np.array([1,2,3,4,5,6])
plt.xticks(x,[0.05, 0.1, 0.2, 0.3, 0.5, 1.0], fontsize=13)
y1 = np.array([0.221, 0.226, 0.2743, 0.265, 0.216, 0.168])
y2 = np.array([0.1426, 0.1446, 0.1647, 0.1613, 0.1446, 0.1348])
y3 = np.array([0.1236, 0.1362, 0.1548, 0.1416, 0.1356, 0.1229])

#plt.xticks(x,[0.05, 0.1, 0.2, 0.5, 0.8, 1.0], fontsize=13)
#y1 = np.array([0.246, 0.2743, 0.265, 0.236, 0.228, 0.204])
#y2 = np.array([0.1546, 0.1647, 0.1613, 0.1546, 0.1348, 0.1247])
#y3 = np.array([0.1462, 0.1548, 0.1516, 0.1316, 0.1229, 0.1189])

#x = np.array([1,2,3,4,5])
#plt.xticks(x,[64, 126, 256, 512, 1024], fontsize=13)
plt.xticks(fontsize=17)
plt.yticks([0.1,0.15,0.2,0.25,0.3],fontsize=17)
# 添加图例

#y1 = np.array([0.198, 0.227, 0.2743, 0.266, 0.232])
#y2 = np.array([0.125, 0.1446, 0.1647, 0.156, 0.1333])
#y3 = np.array([0.1155, 0.1362, 0.1548, 0.142, 0.1206])
plt.ylabel('recall@80',fontsize=20)
plt.plot(x, y1, marker='D', markersize=5.5, label='Beibei')
plt.plot(x, y2, marker='x', markersize=7, label='Taobao')
plt.plot(x, y3, marker='*', markersize=7, label='Yelp')
plt.legend(bbox_to_anchor=(0, 1.02, 1.0, .001), loc='lower left',fontsize=13.5,
           ncol=3, mode="expand", borderaxespad=0.,frameon=False)
#plt.savefig('/Users/gushuyun/Desktop/image/lamda.jpg')
plt.show()
"""









