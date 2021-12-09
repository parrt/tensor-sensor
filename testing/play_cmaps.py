import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mc

bits = [4,8,16,32,64,128]
bits = [8,16,32,64,128]
nhues = len(bits)

blueish = '#3B75AF'
greenish = '#519E3E'

orangeish = '#FDDB7D'
limeish = '#C1E1C5'
limeish = '#A8E1B0'
yellowish = '#FFFFAD'

print(mc.hex2color(limeish))

type_colors = {'float':limeish, 'int':blueish, 'complex':orangeish}

# Derived from https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10

def categorical_cmap(color, nsc):
    # ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    # print(ccolors[0:4])
    cols = np.zeros((nsc, 3))
    # chsv = mc.rgb_to_hsv(c[:3])
    chsv = mc.rgb_to_hsv(mc.hex2color(color))
    arhsv = np.tile(chsv,nsc).reshape(nsc,3)
    arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
    arhsv[:,2] = np.linspace(chsv[2],1,nsc)
    rgb = mc.hsv_to_rgb(arhsv)
    cols[0:nsc,:] = rgb
    cmap = mc.ListedColormap(cols)
    return cmap

plt.figure(figsize=(3,3))
c1 = categorical_cmap(blueish,nhues)
plt.scatter(np.arange(nhues),[1]*nhues, c=np.arange(nhues), s=1080, cmap=c1, linewidths=.5, edgecolors='grey')
c1 = categorical_cmap(limeish,nhues)
plt.scatter(np.arange(nhues),[2]*nhues, c=np.arange(nhues), s=1080, cmap=c1, linewidths=.5, edgecolors='grey')
c1 = categorical_cmap(yellowish,nhues)
plt.scatter(np.arange(nhues),[3]*nhues, c=np.arange(nhues), s=1080, cmap=c1, linewidths=.5, edgecolors='grey')

plt.margins(y=3)
plt.xticks([])
plt.yticks([0,1,2],["(5, 4)", "(2, 5)", "(4, 3)"])
plt.ylim(0, 4)
plt.axis('off')

plt.savefig("/Users/parrt/Desktop/colors.pdf")
plt.show()