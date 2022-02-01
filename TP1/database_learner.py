import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import sys

class Learner:
    # Learn images database
    def learn(self, files):
        for f in files:
            # Get image's name without its extension
            img_name = os.path.basename(f).split('.')[0]
            # Find or create images' index
            if not os.path.exists('index'):
                os.mkdir('index')

            # Image is read as a matrix of pixels
            img_pixels = skimage.io.imread(f)
            # Learn RGB color histograms
            self.learn_rgb(img_name, img_pixels)
            # Learn oriented gradients histograms (HOG)
            self.learn_hog(img_name, img_pixels)

            print('Image ' + img_name + ' learned')
            

    # Learn image's RGB color histogram
    def learn_rgb(self, img_name, img_pixels):
        red_histogram, green_histogram, blue_histogram = self.calc_rgb(img_pixels)

        # Save histograms into the images' index
        file_name = 'index/' + img_name + '_rgb'
        f = open(file_name + '.txt', 'w')
        for r, g, b in zip(red_histogram, green_histogram, blue_histogram):
            f.write(str(r) + '|' + str(g) + '|' + str(b) + '\n')
        f.close()

        # Draw a plot with RGB histograms
        plt.plot(red_histogram, 'r')
        plt.plot(green_histogram, 'g')
        plt.plot(blue_histogram, 'b')
        # Save histograms' plot into the images' index and clear plot
        plt.savefig(file_name + '.png')
        plt.clf()


    # Calculate image's RGB color histogram
    def calc_rgb(self, img_pixels):
        # Convert grayscale images' pixels to RGB color space
        if img_pixels.ndim == 2: # == gray
            img_pixels = skimage.color.gray2rgb(img_pixels)

        # Calculate and return histogram for each color
        red_histogram = np.histogram(img_pixels[:, :, 0], bins = 256)[0]
        green_histogram = np.histogram(img_pixels[:, :, 1], bins = 256)[0]
        blue_histogram = np.histogram(img_pixels[:, :, 2], bins = 256)[0]

        return red_histogram, green_histogram, blue_histogram


    # Learn image's oriented gradients histograms (HOG)
    def learn_hog(self, img_name, img_pixels):
        hog, hog_image = self.calc_hog(img_pixels, True)

        # Save HOG and the image's dimensions into the images' index
        file_name = 'index/' + img_name + '_hog'
        f = open(file_name + '.txt', 'w')
        f.write('|'.join([str(v) for v in hog]) + '\n')
        f.write(str(np.ma.size(hog_image, 1)) + '|' + str(np.ma.size(hog_image, 0)))
        f.close()

        # Save HOG's image into the images' index
        plt.imsave(file_name + '.png', hog_image, cmap = "gray")


    # Calculate image's oriented gradients histograms (HOG)
    def calc_hog(self, img_pixels, calc_img):
        # Convert RGB images' pixels to grayscale color space
        if img_pixels.ndim == 3: # == RGB
            img_pixels = skimage.color.rgb2gray(img_pixels.astype(int))

        # Calculate and return HOG
        return skimage.feature.hog(img_pixels, visualize = calc_img)


# Program main function
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('ERROR: Wrong arguments (see -h)')
    elif sys.argv[1] == '-h':
        help = 'Learns a database of images by calculating their RGB histograms and histograms\n'
        help += 'of oriented gradients (HOG) then saving these into an index.\n'
        help += 'Outputs an index directory containing 4 files per learned image:\n'
        help += '   1. TXT file with HOG containing a first line with the gradient force of each\n'
        help += '      pixel separated by a pipe (|) and a second line with image\'s width|height.\n'
        help += '   2. PNG file illustrating the HOG.\n'
        help += '   3. TXT file with RGB histogram containing 256 lines (one line per bin) with\n'
        help += '      number of pixels having value of bin for R|G|B.\n'
        help += '   4. PNG file illustrating the RGB histogram.\n'
        help += 'Each of these 4 files are prefixed with the image\'s name and suffixed with\n' 
        help += 'the histogram\'s type (RGB or HOG)\n'
        help += '\n'
        help += 'GIVEN DATABASE MUST CONTAIN ONLY IMAGE FILES.\n'
        help += '\n'
        help += 'Usage: database_learner.py database\n'
        help += 'Arguments:\n'
        help += '   database:       Represents the path to the database of images to learn.\n'
        help += '\n'
        help += 'Example:\n'
        help += '   Input: database_learner.py ./database\n'
        help += '   Ouput: index directory containing RGB histograms (TXT & PNG) and HOG (TXT & PNG)\n' 
        help += '          files for each image of the learned database.'

        print(help)
    else:
        database = sys.argv[1]
        if not (os.path.exists(database) and os.path.isdir(database)):
            print('ERROR: Specified database does not exist (see -h)')
        else:
            files = []

            for f in os.listdir(database):
                file_path = os.path.join(database, f)
                file_ext = os.path.splitext(file_path)[-1]
                files = files + [file_path]

            learner = Learner()
            learner.learn(files)