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
            # Image's prefix corresponds to its class
            img_class = img_name.split('_')[0]
            # Find or create image's class in index
            if not os.path.exists('index'):
                os.mkdir('index')
            if not os.path.exists('index/' + img_class):
                os.mkdir('index/' + img_class)

            # Image is read as a matrix of pixels
            img_pixels = skimage.io.imread(f)
            # Learn RGB color histograms
            self.learn_rgb(img_name, img_class, img_pixels)
            # Learn oriented gradients histograms (HOG)
            self.learn_hog(img_name, img_class, img_pixels)


    # Learn image's RGB color histogram
    def learn_rgb(self, img_name, img_class, img_pixels):
        # Convert grayscale images' pixels to RGB color space
        if img_pixels.ndim == 2: # == gray
            img_pixels = skimage.color.gray2rgb(img_pixels)
        # Get red, green and blue tones for each pixel
        red_pixels, green_pixels, blue_pixels = [], [], []
        for pixels_row in img_pixels:
            red_pixels = red_pixels + [p[0] for p in pixels_row]
            green_pixels = green_pixels + [p[1] for p in pixels_row]
            blue_pixels = blue_pixels + [p[2] for p in pixels_row]

        # Calculate histogram for each color
        rgb_range = range(0, 256)
        red_histogram = np.histogram(red_pixels, bins = rgb_range)
        green_histogram = np.histogram(green_pixels, bins = rgb_range)
        blue_histogram = np.histogram(blue_pixels, bins = rgb_range)
        # Save histograms into the image's class index
        file_name = 'index/' + img_class + '/' + img_name + '_rgb'
        f = open(file_name + '.txt', 'w')
        for rp, gp, bp in zip(red_histogram[0], green_histogram[0], blue_histogram[0]):
            f.write(str(rp) + '|' + str(gp) + '|' + str(bp) + '\n')
        f.close()

        # Draw a plot with RGB histograms
        plt.plot(red_histogram[0], 'r')
        plt.plot(green_histogram[0], 'g')
        plt.plot(blue_histogram[0], 'b')
        # Save histograms' plot into the image's class index and clear plot
        plt.savefig(file_name + '.png')
        plt.clf()


    # Learn image's oriented gradients histograms (HOG)
    def learn_hog(self, img_name, img_class, img_pixels):
        # Convert RGB images' pixels to grayscale color space
        if img_pixels.ndim == 3: # == RGB
            img_pixels = skimage.color.rgb2gray(img_pixels)

        # Calculate HOG
        hog, hog_image = skimage.feature.hog(img_pixels, visualize = True)

        # Save HOG into the image's class
        file_name = 'index/' + img_class + '/' + img_name + '_hog'
        f = open(file_name + '.txt', 'w')
        f.write('|'.join([str(v) for v in hog]))
        f.close()

        # Save HOG's image into the image's class index
        plt.imsave(file_name + '.png', hog_image, cmap = "gray")


# Program main function
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('ERROR: Wrong arguments (see -h)')
    elif sys.argv[1] == '-h':
        help = 'TODO'
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