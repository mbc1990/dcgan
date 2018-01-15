import os
import imageio

gif_name = "out_150x150x4.gif"

def cmp(x):
    spl = x.split("_")
    return int(spl[1].split(".png")[0])


def main():
    dir = "output/"
    files = os.listdir(dir)
    files.sort(key=cmp)
    for_gif = []
    files = ["output/" + f for f in files]
    for idx, file in enumerate(files):
        if idx % 5 == 0:
            for_gif.append(file)
    create_gif(for_gif, 0)


def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = gif_name
    imageio.mimsave(output_file, images, duration=duration)    


if __name__ == "__main__":
    main()
