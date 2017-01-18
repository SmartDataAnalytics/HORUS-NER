import os

import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

if 1==2:
    objects = ('Coast', 'Forest', 'Highway', 'Mountain', 'Open Country', 'Inside City', 'Street', 'Suburb', 'Tall Building')
    y_pos = np.arange(len(objects))
    performance = [0.89, 0.96, 0.83, 0.84, 0.88, 0.89, 0.87, 0.89, 0.90]

    plt.barh(y_pos, performance, align='center', alpha=0.5, color='red')
    plt.yticks(y_pos, objects)
    plt.xlabel('Accuracy')
    plt.title('Location-based classifiers')

    plt.show()
'''
======================================================= ALL =======================================================
'''

place_acc = (0.89 + 0.96 + 0.83 + 0.84 + 0.88 + 0.89 + 0.87 + 0.89 + 0.90) / 9

place_tp = (0.81 + 1.0 + 0.84 + 0.94 + 0.92 + 0.94 + 0.92 + 0.96 + 0.92) / 9
print place_tp
place_fp = (0.04 + 0.08 + 0.18 + 0.26 + 0.16 + 0.16 + 0.18 + 0.18 + 0.12) / 9
print place_fp
place_fn = (0.19 + 0.00 + 0.16 + 0.06 + 0.08 + 0.06 + 0.08 + 0.04 + 0.08) / 9
print place_fn


# tp / (tp + fp)
p_place = place_tp / (place_tp + place_fp)
# tp / (tp + fn)
r_place = place_tp / (place_tp + place_fn)
# 2 * (precision * recall) / (precision + recall)
f1_place = 2 * (p_place * r_place) / (p_place + r_place)

p_logo = 0.88 / (0.88 + 0.00)
r_logo = 0.88 / (0.88 + 0.12)
f1_logo = 2 * (p_logo * r_logo) / (p_logo + r_logo)

p_face = 0.96 / (0.96 + 0.26)
r_face = 0.96 / (0.96 + 0.04)
f1_face = 2 * (p_face * r_face) / (p_face + r_face)


x = ['Place', 'Logo', 'Face']
y_pos = np.arange(len(x))
y = [f1_place, f1_logo, f1_face]

fig, ax = plt.subplots()
width = 0.50 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('HORUS Computer Vision Modules')
plt.xlabel('F1 measure')
plt.ylabel('module')
for i, v in enumerate(y):
    print i, v
    ax.text(v + 0.05, i + .3, "{0:.2f}".format(v), color='blue', fontweight='bold', ha='center', va='center')
#plt.show()
plt.savefig(os.path.join('test.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
