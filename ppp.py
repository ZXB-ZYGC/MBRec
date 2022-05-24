import numpy as np
import matplotlib.pyplot as plt

size = 2
x = np.arange(size)

labelsize = 28
ticksize = 28
legendsize = 20

'''
v1 = np.array([67.4, 70.1, 89.9])
v1_es = np.array([67.8,71.3,90.8])
v2 = np.array([67.9, 71.0, 90.5])
v2_es = np.array([68.3,71.4,90.7])
all = np.array([70.2,73.4,91.4])
'''
v1 = np.array([67.4, 70.1])
v1_es = np.array([67.8,71.3])
v2 = np.array([67.9, 71.0])
v2_es = np.array([68.3,71.4])
all = np.array([70.2,73.4])

total_width, n = 1.0, 6
width = total_width / n -0.07
x = x - (total_width - width) / 2
print(x + 2 * (width+0.03))

fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.xticks(x + 2 * (width+0.03), ["F1-macro","F1-micro"],fontsize=labelsize)
plt.yticks([66,68,70,72,74],["0.66","0.68","0.70","0.72","0.74"],fontsize=ticksize)
plt.ylim([65.9,75])
plt.ylabel("Evaluation", fontsize=labelsize)

plt.bar(x, v1,  width=width, label='V1_ori', color='salmon', zorder=100)
plt.bar(x + width+0.03, v1_es, width=width, label='V1_es', color='skyblue', zorder=100)
plt.bar(x + 2 * (width+0.03), v2, width=width, label='V2_ori', color='tan', zorder=100)
plt.bar(x + 3 * (width+0.03), v2_es, width=width, label='V2_es', color='lightpink', zorder=100)
plt.bar(x + 4 * (width+0.03), all, width=width, label='CoGSL', color='darkseagreen', zorder=100)
#plt.legend(loc="upper left",fontsize=fontsize_tu)#prop={'size':8},
plt.legend(bbox_to_anchor=(.0, 1.02, 1.1, .001), loc='lower left',fontsize=legendsize, bbox_transform=ax1.transAxes,
           ncol=3, mode="expand", borderaxespad=0.,frameon=False)
plt.grid(axis="y", alpha=0.8, linewidth=1.5, zorder=0)
plt.savefig("xiao_citeseer.png",bbox_inches='tight',dpi=300,pad_inches=0.1)
plt.show()




import matplotlib.pyplot as plt
import numpy as np

dataset = "citeseer"
term = "macro"

xticksize = 20
yticksize = 20
labelsize = 28
textsize = 15

data = np.array([69.2,	69.3,	70,	70.1,	70.15,	70.2,	69.7,	68.8,	69.2])
data_txt = ["0.692","0.693","0.700","0.701","0.7015","0.702","0.697","0.688","0.692"]
ylim = [68.7,70.5]
yticks = [68.8,69.1,69.4,69.7,70,70.3]
yticklabels = ["0.688","0.691","0.694","0.697","0.700","0.703",]

x=[0,1,2,3,4,5,6,7,8]
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.set_xlabel('The value of η', fontsize=labelsize)
axes.set_xticks(x)
axes.set_xticklabels(["0.0","1e-9","1e-7","1e-5","1e-2","5e-2","1e-1","1e1","1e2"], fontsize=xticksize, rotation=50)
axes.set_xlim([-1, 9])

axes.set_ylabel('Test F1-'+term, fontsize=labelsize)
axes.set_yticks(yticks)
axes.set_yticklabels(yticklabels, fontsize=yticksize)
axes.set_ylim(ylim)

mini = data.min()
diff = data-mini

ll = np.arange(len(data))
ll = -np.sort(-ll)
s = 50+ll*50

d= 5-diff
axes.scatter(x, data,s=s,c=d,cmap="copper_r", marker="o", zorder=100)
axes.scatter(np.argmax(data), data.max(), marker="*", c="red", s=90, zorder=100)

juli = ll*0.001
juli = juli+0.06
print(juli)
for i in range(len(data)):
    plt.annotate(data_txt[i],xy=(x[i], data[i]), xytext=(x[i]-0.8, data[i]+juli[i]), fontsize=textsize)

plt.grid(alpha=0.8, linewidth=1.5, zorder=0)

plt.savefig("mi_"+dataset+"_"+term+".png",bbox_inches='tight',dpi=500,pad_inches=0.1)
plt.show()

