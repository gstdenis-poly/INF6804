import os
import argparse

import torch
import cv2

from test import GOTURN

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/OTB/Man', type=str,
                    help='path to video frames')
parser.add_argument('-s', '--save-directory',
                    default='../result',
                    type=str, help='path to save directory')
parser.add_argument('-i', '--index',
                    default=0,
                    type=int, help='index of starting frame')
parser.add_argument('-l', '--length',
                    default=800,
                    type=int, help='number of frames')


def axis_aligned_iou(boxA, boxB):
    # make sure that x1,y1,x2,y2 of a box are valid
    assert(boxA[0] <= boxA[2])
    assert(boxA[1] <= boxA[3])
    assert(boxB[0] <= boxB[2])
    assert(boxB[1] <= boxB[3])

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def save(im, bb, idx, f):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    bb = [int(val) for val in bb]  # GOTURN output
    # plot GOTURN predictions with red rectangle
    im = cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]),
                       (0, 0, 255), 2)
    save_path = os.path.join(args.save_directory, str(idx)+'.jpg')
    cv2.imwrite(save_path, im)
    # save bounding box into results.txt
    bb_width = abs(bb[0] - bb[2])
    bb_height = abs(bb[1] - bb[3])
    f.write(str(idx) + ' ' + str(bb[0]) + ' ' + str(bb[1]) + ' ')
    f.write(str(bb_width) + ' ' + str(bb_height) + '\n')


def main(args):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    tester = GOTURN(args.data_directory, args.index, args.length,
                    args.model_weights,
                    device)
    if os.path.exists(args.save_directory):
        print('Save directory %s already exists' % (args.save_directory))
    else:
        os.makedirs(args.save_directory)

    # Initiate results.txt file
    f = open(args.save_directory + '/results.txt', 'w')

    # save initial frame with bounding box
    save(tester.img[0][0], tester.prev_rect, 1, f)
    tester.model.eval()

    # loop through sequence images
    for i in range(tester.idx, tester.len):
        # get torch input tensor
        sample = tester[i]

        # predict box
        bb = tester.get_rect(sample)
        tester.prev_rect = bb

        # save current image with predicted rectangle
        im = tester.img[i][1]
        save(im, bb, i+2, f)

    f.close() # close results.txt


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
