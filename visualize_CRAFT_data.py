import os
import cv2
from h11 import Data
import matplotlib.pyplot as plt

DATA = 'CRAFT/data_root_dir'
training_images = 'ch4_training_images'
training_gt = 'ch4_training_localization_transcription_gt'

# DATA = 'processing/KAIST'
# training_images = 'images'
# training_gt = 'localization_transcription_gt'

def main():
    for filename in os.listdir(os.path.join(DATA, training_images)):
        text, ext = os.path.splitext(filename)

        if ext == '.jpg':
            img = cv2.imread(os.path.join(DATA, training_images, filename))
            img_y, img_x, _ = img.shape

            with open(os.path.join(DATA, training_gt, f'gt_{text}.txt'), 'r') as f:
                lines = f.readlines()
            xlist = []
            ylist = []
            for line in lines:
                x = []
                y = []
                line = line.split(',')
                x.append(int(line[0]))
                x.append(int(line[2]))
                x.append(int(line[4]))
                x.append(int(line[6]))
                x.append(int(line[0]))
                y.append(int(line[1]))
                y.append(int(line[3]))
                y.append(int(line[5]))
                y.append(int(line[7]))
                y.append(int(line[1]))
                xlist.append(x)
                ylist.append(y)

            plt.cla()
            plt.imshow(img)
            for i in range(len(xlist)):
                plt.plot(xlist[i], ylist[i])
            plt.xlim(0, img_x)
            plt.ylim(img_y, 0)
            plt.show(block=False)
        
        if input('Continue? [y/n] ') == 'n':
            break

if __name__ == '__main__':
    main()