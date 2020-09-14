import matplotlib
import matplotlib.patches as patches
from matplotlib import pyplot as plt
def textdim(s, fontsize=11):
    fig, ax = plt.subplots(1,1)
    t = ax.text(0, 0, s, bbox={'lw':0}, fontsize=fontsize)
    plt.savefig("/tmp/junk")
    plt.close()
    bb = t.get_bbox_patch()
    w, h = bb.get_width(), bb.get_height()
    return w, h

# print(textdim("test of foo", fontsize=11))
# print(textdim("test of foO", fontsize=11))
# print(textdim("W @ b + x * 3 + h.dot(h)", fontsize=12))

code = 'W  @  b + x * 3 + h.dot(h)'

w,h = textdim(code,fontsize=12)
print(w,h)
fig, ax = plt.subplots(1,1, figsize=(w/200,h/200))

ax.text(0,0,code,fontsize=12,fontname='Consolas')

ax.set_xlim(0,w)
ax.set_ylim(0,h)

ax.axis('off')
#plt.tight_layout()

plt.savefig("/tmp/t.pdf", bbox_inches='tight', pad_inches=.01, dpi=200)
