from PIL import Image
import os
import argparse

#Reference Template:   Name, Date, Title of Program, Code Version, Web Adrress
#Reference: Gilbert Tanner, N/A, transform_image_resolution.py, N/A, https://gilberttanner.com/blog/creating-your-own-objectdetector

def rescale_images(directory, size):
    for img in os.listdir(directory):
        im = Image.open(directory+img)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(directory+img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the images')
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
    args = parser.parse_args()
    rescale_images(args.directory, args.size)