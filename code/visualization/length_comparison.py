import numpy as np
import numpy as np

import matplotlib.pyplot as plt

# 构建示例数据

# set width of bars
barWidth = 0.10
Font_title = {'size': 12, 'family': 'Times New Roman'}

# set heights of bars
bars1 = [0.8892, 0.9404, 0.9720]
bars2 = [0.8892, 0.9404, 0.9720]
bars3 = [0.8892, 0.9404, 0.9720]
bars4 = [0.8892, 0.9404, 0.9720]
bars5 = [0.8882, 0.9309, 0.9673]
bars6 = [0.9590, 0.9855, 0.9956]
bars7 = [0.9597, 0.9859, 0.9957]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]

# 绘制分组条形图
plt.figure(figsize=(10, 10))
# Make the plot
plt.bar(r1, bars1, color='aqua', width=barWidth, edgecolor='white', label='Accuracy')
plt.bar(r2, bars2, color='steelblue', width=barWidth, edgecolor='white', label='F1-score')
plt.bar(r3, bars3, color='beige', width=barWidth, edgecolor='white', label='Precision')
plt.bar(r4, bars4, color='coral', width=barWidth, edgecolor='white', label='Recall')
plt.bar(r5, bars5, color='crimson', width=barWidth, edgecolor='white', label='Specificity')
plt.bar(r6, bars6, color='green', width=barWidth, edgecolor='white', label='AUC')
plt.bar(r7, bars7, color='silver', width=barWidth, edgecolor='white', label='AUPR')

# Add xticks on the middle of the group bars
plt.xlabel('negative sample length', Font_title)
plt.ylabel('value', Font_title)
plt.xticks([r + barWidth*3 for r in range(len(bars1))], ['1X', '1.5X', '2X'])
plt.tick_params(labelsize=10)

# Create legend & Show graphic
#plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(-0.03,1.05), ncol=7)
plt.show()