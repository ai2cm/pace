from PIL import Image
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir("./plot_output") if isfile(join("./plot_output", f))]
onlyfiles.sort()
imgs = []
for f in onlyfiles:
    imgs.append(Image.open(f"./plot_output/{f}"))


imgs.pop(0).save("out.gif", save_all=True, append_images=imgs, duration=200, loop=1)
