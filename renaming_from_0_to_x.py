import os
imdir = 'C:/Users/udomc/anaconda3/yolov4-custom-functions-master/Globlex_dataset_2/Tao'
#renaming each image from the imdir to be from 0 to 9
#Reference Template:   Name, Date, Title of Program, Code Version, Web Adrress
#Reference: Khanin Udomchoksakul, December 16th, 2019, renaming_from_0_to_x.py, 3r revision
n = 0
for imfile in os.scandir(imdir):
    os.rename(imfile.path, os.path.join(imdir, '{:06}.jpg'.format(n)))

    n += 1
