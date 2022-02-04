from database_learner import Learner
import numpy as np
import os
import scipy
import skimage
import sys


class Util:
    @staticmethod # Adjust image to given dimensions
    def adjust_img(img_name, img, dimensions, keep_proportions):
        adjusted_img = img
        if keep_proportions:
            adjusted_img = Util.pad_img(img, dimensions)
        else:
            adjusted_img = Util.resize_img(img, dimensions)

        # Save adjusted image into an output directory
        if not os.path.exists('output'):
            os.mkdir('output')

        suffix = 'pad' if keep_proportions else '' # Suffix to output image file
        skimage.io.imsave('output/' + img_name + '_' + suffix + '.png', adjusted_img)

        return adjusted_img


    @staticmethod # Resize and pad given image to given dimensions
    def pad_img(img, dimensions):
        adjusted_img = img
        is_rgb = adjusted_img.ndim == 3

        # Reduce size of img if bigger than given dimensions
        img_width = np.ma.size(img, 1)
        img_height = np.ma.size(img, 0)
        if dimensions[0] < img_width or dimensions[1] < img_height:
            width_scale = dimensions[0] / img_width
            height_scale = dimensions[1] / img_height
            scale = width_scale if width_scale < height_scale else height_scale
            # Scale has 3 dimensions for RGB and 2 for grayscale
            scale = (scale, scale, 1) if is_rgb else (scale, scale)
            adjusted_img = skimage.transform.rescale(img, scale, 
                                                     preserve_range = True, 
                                                     anti_aliasing = True)
        adjusted_img_width = np.ma.size(adjusted_img, 1)
        adjusted_img_height = np.ma.size(adjusted_img, 0)

        # Horizontal padding
        diff_h = abs(adjusted_img_width - dimensions[0]) # Difference between query and ref widths
        # Padding has 3 dimensions for RGB and 2 for grayscale
        padding_h = ((0, 0), (0, diff_h), (0, 0)) if is_rgb else ((0, 0), (0, diff_h))
        if adjusted_img_width < dimensions[0]:
            adjusted_img = np.pad(adjusted_img, padding_h)
        # Vertical padding
        diff_v = abs(adjusted_img_height - dimensions[1])
        # Padding has 3 dimensions for RGB and 2 for grayscale
        padding_v = ((0, diff_v), (0, 0), (0, 0)) if is_rgb else ((0, diff_v), (0, 0))
        if adjusted_img_height < dimensions[1]:
            adjusted_img = np.pad(adjusted_img, padding_v)

        return adjusted_img


    @staticmethod # Resize image to given dimensions
    def resize_img(img, dimensions):
        adjusted_img = skimage.transform.resize(img, 
                                                (dimensions[1], dimensions[0]), 
                                                anti_aliasing = True)
        # Convert the image to a 0-255 scale.
        adjusted_img = 255 * adjusted_img
        # Convert to integer data type pixels.
        adjusted_img = adjusted_img.astype(np.uint8)

        return adjusted_img


class Processor:
    index = {}
    keep_proportions = False
    use_thresholds = False

    def __init__(self, files, keep_proportions, use_thresholds):
        self.keep_proportions = keep_proportions
        self.use_thresholds = use_thresholds
        for f in files:
            # Get file's name without its extension
            file_name = os.path.basename(f).split('.')[0]
            # Get histogram's type of file by its suffix
            hist_type = file_name.split('_')[-1] if '_' in file_name else None
            if hist_type == None:
                continue
            
            # Get image's name by removing histogram's type in file's name
            img_name = file_name.split('_' + hist_type)[0] 
            if img_name not in self.index:
                self.index[img_name] = {}

            f_raw = open(f, 'r').read()
            if hist_type == 'rgb':
                histogram_bins = f_raw.split('\n')[:-1] # Ignore empty last value
                histogram = []
                for bar in histogram_bins:
                    histogram += [[int(v) for v in bar.split('|')]]

                self.index[img_name]['rgb'] = np.array(histogram)
            elif hist_type == 'hog':
                f_lines = f_raw.split('\n')
                histogram = [float(v) for v in f_lines[0].split('|')]
                img_dimensions = [int(v) for v in f_lines[1].split('|')]
                self.index[img_name]['hog'] = np.array(histogram)
                self.index[img_name]['dimensions'] = img_dimensions


    # Process CBIR
    def process(self, query):
        # Get query image's name without its extension
        query_img_name = os.path.basename(query).split('.')[0]
        # Get query's prefix as the query's class
        query_class = query_img_name.split('_')[0]
        # Query's image is read as a matrix of pixels
        query_pixels = skimage.io.imread(query)
        # Calculate query image's RGB histograms because invariant to images' size
        learner = Learner()
        r, g, b = learner.calc_rgb(query_pixels)
        query_histograms = { 'rgb': np.vstack((r, g, b)).T }
        # Calculate feature vector comparisons
        l1_norms, l2_norms, bhattacharyya, mdpa, cosine = {}, {}, {}, {}, {}
        for img in self.index:
            ref = self.index[img] # histograms of image in index
            # Calculate query image's HOG because variable with images' size
            hog = learner.calc_hog(Util.adjust_img(query_img_name + '-' + img,
                                                   query_pixels, 
                                                   ref['dimensions'],
                                                   self.keep_proportions), False)
            query_histograms['hog'] = np.array(hog)
            del self.index[img]['dimensions'] # Not needed anymore
            # Compare query with reference using different feature vector comparison
            l1_norms[img] = self.process_lp_norms(query_histograms, ref, 1)
            l2_norms[img] = self.process_lp_norms(query_histograms, ref, 2)
            bhattacharyya[img] = self.process_bhattacharyya(query_histograms, ref)
            mdpa[img] = self.process_mdpa(query_histograms, ref)
            cosine[img] = self.process_cosine(query_histograms, ref)
        # Print output of CBIR
        print('\nImages retrieved based on content (CBIR):\n')
        for histogram in query_histograms:
            self.print_result(query_class, l1_norms, histogram, 'l1 norms')
            self.print_result(query_class, l2_norms, histogram, 'l2 norms')
            self.print_result(query_class, bhattacharyya, histogram, 'Bhattacharyya distance')
            self.print_result(query_class, mdpa, histogram, 'Maximum difference of pair assignments')
            self.print_result(query_class, cosine, histogram, 'Cosine similarity')


    # Print result of CBIR
    def print_result(self, query_class, result, histogram, comparison):
        # Sort images by ascending value of comparison
        sorted_result = sorted(result, key = lambda k: result[k][histogram])

        true_positives_counter = 0
        expected_positives_counter = 0
        average_precision_3 = 0
        average_precision_5 = 0

        print(histogram.upper() + ' ' + comparison + ':')
        for img in sorted_result:
            distance = result[img][histogram]
            if not self.use_thresholds or \
               histogram == 'rgb' and comparison == 'Bhattacharyya distance' and distance < 0.26 or \
               histogram == 'rgb' and comparison == 'Cosine similarity' and distance < 0.08 or \
               histogram == 'hog' and comparison == 'Bhattacharyya distance' and distance < 0.44 or \
               histogram == 'hog' and comparison == 'Cosine similarity' and distance < 0.71:
                print(img + ': ' + str(distance))
            
            expected_positives_counter += 1
            current_recall = 0
            if query_class == img.split('_')[0]: # Query and index images have same class
                true_positives_counter += 1
                current_recall = true_positives_counter / expected_positives_counter

            if expected_positives_counter <= 5:
                average_precision_5 += current_recall
            if expected_positives_counter <= 3:
                average_precision_3 += current_recall

        average_precision_5 = (1 / 5) * average_precision_5
        average_precision_3 = (1 / 3) * average_precision_3

        print('Performance of CBIR query for top 5 retrieved images: ' + str(average_precision_5))
        print('Performance of CBIR query for top 3 retrieved images: ' + str(average_precision_3))
        print('\n')


    # Calculate lp norm of given order (1 or 2)
    def process_lp_norms(self, query, reference, ord):
        lp_norms = {}
        for histogram in query:
            histograms_distance = query[histogram] - reference[histogram]
            lp_norms[histogram] = np.linalg.norm(histograms_distance, ord = ord)

        return lp_norms


    # Calculate Bhattacharyya distance
    def process_bhattacharyya(self, query, reference):
        bhattacharyya = {}
        for histogram in query:
            # Normalize histograms
            ref_histogram = reference[histogram] / np.sum(reference[histogram])
            query_histogram = query[histogram] / np.sum(query[histogram])
            # Calculate Bhattacharyya coefficient and distance
            coefficient = np.sum(np.sqrt(np.multiply(ref_histogram, query_histogram)))
            bhattacharyya[histogram] = np.log(coefficient) * -1

        return bhattacharyya


    # Calculate Maximum difference of pair assignments (Wasserstein distance)
    def process_mdpa(self, query, reference):
        mdpa = {}
        for histogram in query:
            # Convert Numpy histograms to Python lists for Scipy
            ref_histogram = reference[histogram].flatten().tolist()
            query_histogram = query[histogram].flatten().tolist()

            mdpa[histogram] = scipy.stats.wasserstein_distance(ref_histogram, query_histogram)

        return mdpa


    # Calculate cosine distance
    def process_cosine(self, query, reference):
        cosine = {}
        for histogram in query:
            # Convert Numpy histograms to Python lists for Scipy
            ref_histogram = reference[histogram].flatten().tolist()
            query_histogram = query[histogram].flatten().tolist()

            cosine[histogram] = scipy.spatial.distance.cosine(ref_histogram, query_histogram)

        return cosine


# Program main function
if __name__ == '__main__':
    if len(sys.argv) < 2 or 4 < len(sys.argv):
        print('ERROR: Wrong arguments (see -h)')
    elif len(sys.argv) == 2 and sys.argv[1] == '-h':
        help = 'Processes a CBIR query by computing the RGB histograms and histograms of oriented\n' 
        help += 'gradients (HOG) of a given image and comparing them with an index of histograms\n'
        help += 'from previously learned images.\n'
        help += 'Outputs images of the index sorted by highest similarity to query image\n'
        help += 'according to the 5 following feature vector comparison methods applied on the\n'
        help += 'RGB histograms and HOG:\n'
        help += '   1. l1 norm.\n'
        help += '   2. l2 norm.\n'
        help += '   3. Bhattacharyya distance\n'
        help += '   4. Maximum difference of pair assignments.\n'
        help += '   5. Cosine similarity.\n'
        help += '\n'
        help += 'A DIRECTORY NAMED "index" CONTAINING THE HISTOGRAMS (RGB & HOG) IN TXT FILES\n'
        help += 'MUST EXIST IN THE SAME DIRECTORY AS THIS QUERY PROCESSOR. THE HISTOGRAM FILES\n'
        help += 'MUST RESPECT THE FOLLOWING FORMAT:\n'
        help += '   - HOG: FIRST LINE WITH THE GRADIENT FORCE OF EACH PIXEL SEPARATED BY A\n' 
        help += '          PIPE (|) AND A SECOND LINE WITH IMAGE\'S WIDTH|HEIGHT.\n'
        help += '   - RGB: 256 LINES (ONE LINE PER BIN) WITH NUMBER OF PIXELS HAVING VALUE OF\n' 
        help += '          BIG FOR R|G|B.256 lines (one line per bin) with number of pixels\n' 
        help += '          having value of bin for R|G|B.\n' 
        help += '\n'
        help += 'Usage: query_processor.py image\n'
        help += 'Arguments:\n'
        help += '   image:       Represents the path to the image to use for CBIR query.\n'
        help += '   -p:          Indicates if proportions of the image must be kept. If this\n'
        help += '                argument is specified, the given image is resized then padded\n'
        help += '                to fit the size of the compared database images. If not\n'
        help += '                specified, the image is resized to fit the size of the compared\n'
        help += '                database images even if it breaks its proportions.\n'
        help += '   -t           Indicates if thresholds must be used to avoid returning images\n'
        help += '                that are not enough similar according to fixed thresholds.\n'
        help += '\n'
        help += 'Example (assuming the index contains the histograms of only cat_2 and cat_5 images):\n'
        help += '   Input: query_processor.py ./airplane_query.jpg\n'
        help += '   Output 1: output directory containing all adjusted images for HOG calculation.\n'
        help += '   Output 2:\n'
        help += '      Images retrieved based on content (CBIR):\n'
        help += '\n'
        help += '      RGB l1 norms:\n'
        help += '      cat_5: 55148.0\n'
        help += '      cat_2: 60300.0\n'
        help += '\n'
        help += '      RGB l2 norms:\n'
        help += '      cat_5: 9738.092645082508\n'
        help += '      cat_2: 9744.219832842178\n'
        help += '\n'
        help += '      RGB Bhattacharyya distance:\n'
        help += '      cat_2: 0.10373204829349397\n'
        help += '      cat_5: 0.11974314957985487\n'
        help += '\n'
        help += '      RGB Maximum difference of pair assignments:\n'
        help += '      cat_5: 81.84375000000004\n'
        help += '      cat_2: 109.35677083333323\n'
        help += '\n'
        help += '      RGB Cosine similarity:\n'
        help += '      cat_2: 0.28385054567163737\n'
        help += '      cat_5: 0.2859208568416892\n'
        help += '\n'
        help += '      HOG l1 norms:\n'
        help += '      cat_5: 5775.877028823786\n'
        help += '      cat_2: 7631.383241007096\n'
        help += '\n'
        help += '      HOG l2 norms:\n'
        help += '      cat_5: 30.16620582302731\n'
        help += '      cat_2: 32.40370295600993\n'
        help += '\n'
        help += '      HOG Bhattacharyya distance:\n'
        help += '      cat_2: 0.36224831300733795\n'
        help += '      cat_5: 0.4825202758847199\n'
        help += '\n'
        help += '      HOG Maximum difference of pair assignments:\n'
        help += '      cat_5: 0.078359476546201\n'
        help += '      cat_2: 0.08972819792950215\n'
        help += '\n'
        help += '      HOG Cosine similarity:\n'
        help += '      cat_2: 0.6571360299854354\n'
        help += '      cat_5: 0.6881324134050362'

        print(help)
    else:
        query = sys.argv[1]
        if not os.path.exists(query) or os.path.isdir(query):
            print('ERROR: Query is not a file (see -h)')
        elif not (os.path.exists('index') and os.path.isdir('index')):
            print('ERROR: Index does not exist (see -h)')
        else:
            files = []

            for f in os.listdir('index'):
                file_path = os.path.join('index', f)
                file_ext = os.path.splitext(file_path)[-1]
                
                if file_ext != '.txt': # Only consider .txt files
                    continue

                files = files + [file_path]

            keep_proportions = len(sys.argv) == 3 and sys.argv[2] == '-p' or \
                               len(sys.argv) == 4 and (sys.argv[2] == '-p' or sys.argv[3] == '-p')
            use_thresholds = len(sys.argv) == 3 and sys.argv[2] == '-t' or \
                             len(sys.argv) == 4 and (sys.argv[2] == '-t' or sys.argv[3] == '-t')
            processor = Processor(files, keep_proportions, use_thresholds)
            processor.process(query)