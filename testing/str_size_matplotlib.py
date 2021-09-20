import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib import pyplot as plt


def textdim(s, fontname="Consolas", fontsize=11):
    fig, ax = plt.subplots(1, 1)
    t = ax.text(0, 0, s, bbox={"lw": 0, "pad": 0}, fontname=fontname, fontsize=fontsize)
    # plt.savefig(tempfile.mktemp(".pdf"))
    plt.savefig("/tmp/font.pdf", pad_inches=0, dpi=200)
    print(t)
    plt.close()
    bb = t.get_bbox_patch()
    print(bb)
    w, h = bb.get_width(), bb.get_height()
    return w, h


# print(textdim("@"))
# exit()


# From: https://stackoverflow.com/questions/22667224/matplotlib-get-text-bounding-box-independent-of-backend
def find_renderer(fig):
    if hasattr(fig.canvas, "get_renderer"):
        # Some backends, such as TkAgg, have the get_renderer method, which
        # makes this easy.
        renderer = fig.canvas.get_renderer()
    else:
        # Other backends do not have the get_renderer method, so we have a work
        # around to find the renderer.  Print the figure to a temporary file
        # object, and then grab the renderer that was used.
        # (I stole this trick from the matplotlib backend_bases.py
        # print_figure() method.)
        import io

        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return renderer


def textdim(s, fontname="Consolas", fontsize=11):
    fig, ax = plt.subplots(1, 1)
    t = ax.text(0, 0, s, fontname=fontname, fontsize=fontsize, transform=None)
    bb = t.get_window_extent(find_renderer(fig))
    print(s, bb.width, bb.height)

    # t = mpl.textpath.TextPath(xy=(0, 0), s=s, size=fontsize, prop=fontname)
    # bb = t.get_extents()
    # print(s, "new", bb)
    plt.close()
    return bb.width, bb.height


# print(textdim("test of foo", fontsize=11))
# print(textdim("test of foO", fontsize=11))
# print(textdim("W @ b + x * 3 + h.dot(h)", fontsize=12))

code = "W@ b + x *3 + h.dot(h)"
code = "W@ b.f(x,y)"  # + x *3 + h.dot(h)'

fig, ax = plt.subplots(1, 1, figsize=(4, 1))

fontname = "Serif"
fontsize = 16

# for c in code:
#     t = ax.text(0,0,c)
#     bbox1 = t.get_window_extent(find_renderer(fig))
#     # print(c, '->', bbox1.width, bbox1.height)
#     print(c, '->', textdim(c, fontname=fontname, fontsize=fontsize))
# rect1 = patches.Rectangle((0,0), bbox1.width, bbox1.height, \
#     color = [0,0,0], fill = False)
# fig.patches.append(rect1)

x = 0
for c in code:
    # print(f"plot {c} at {x},{0}")
    ax.text(x, 10, c, fontname=fontname, fontsize=fontsize, transform=None)
    w, h = textdim(c, fontname=fontname, fontsize=fontsize)
    # print(w,h,'->',x)
    x = x + w

ax.set_xlim(0, x)
# ax.set_ylim(0,10)

ax.axis("off")

# plt.show()

plt.tight_layout()

plt.savefig("/tmp/t.pdf", bbox_inches="tight", pad_inches=0, dpi=200)
