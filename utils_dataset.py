import os
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
from shutil import copyfile, copy


def nifti_to_jpg(path, filename, slice_dim, save_folder):
    """
    Get a slice from nifti image, transform and save it as jpg
    """
    # get path
    filepath = os.path.join(path, filename)

    # load image
    img = nib.load(filepath)

    """
    # The complete information embedded in an image header is available via a format-specific header object.
    hdr = img.header
    print(hdr)
    """

    # image as ndarray
    data = img.get_fdata()

    # get a 2D slice, ixi2 dataset
    # make a slice on slice_dim position
    if data.shape[2] > 100:
        image_slice = data[:, :, slice_dim]
    # skip image, if not enough dimensions
    else:
        return

    """
    Normalize image, adjust pixel values distribution [0 ... 65535] -> [0 ... 255].
    optional: create and show histogram and cumulative histogram
    """
    #grayscale = image_slice / 256
    grayscale = cv2.normalize(image_slice, None, 0, 255, cv2.NORM_MINMAX)

    # optional: create and show histogram and cumulative histogram
    # visualize pixel values distribution of the image
    hist, bins = np.histogram(grayscale.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(grayscale.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    # save new image
    filename_new = filename.split('.')[0] + "-" + str(slice_dim) + "-" + save_folder
    save_path = os.path.join('data', save_folder, filename_new)
    cv2.imwrite(save_path + '.jpg', grayscale)
    return


def extract_multiple(path):
    """
    Extract 2D image from every 3D image in some folder
    """
    files = os.listdir(path)
    # fitting dimensions for ixi2 dataset around 65-80
    slice_dim = 75
    save_folder = "normalized"
    for filename in files:
        nifti_to_jpg(path, filename, slice_dim, save_folder)
    print("complete")
    return


def extract_one():
    """
    Extract 2D image from one 3D image
    """
    path = os.path.join(os.getcwd(), 'data\\ixi2')
    # filename = "IXI035-IOP-0873-T2.nii.gz"
    filename = "IXI002-Guys-0828-T2.nii.gz"
    # fitting dimensions for ixi2 dataset around 65-80
    slice_dim = 60
    save_folder = "normalized"
    nifti_to_jpg(path, filename, slice_dim, save_folder)
    print("complete")
    return


def resize(path, save_path):
    """
    Resize images
    """
    files = os.listdir(path)
    size = (32, 32)
    for filename in files:
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        result = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(save_path, filename), result)
    print("complete")
    return


def crop_images(path, save_path):
    """
    Cut black regions on the sides of the image
    """
    files = os.listdir(path)

    for filename in files:
        # read image
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        # find contours in image
        ret, thresh = cv2.threshold(img, 20, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            # find the biggest contour by the area
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            cropped_img = img[y:y + h, x:x + w]
            # save cropped image
            cv2.imwrite(os.path.join(save_path, filename), cropped_img)

    print("complete")
    return


def rotate(path, save_path):
    """
    Rotate images
    """
    files = os.listdir(path)
    for filename in files:
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        result = cv2.rotate(img, cv2.cv2.ROTATE_180)
        cv2.imwrite(os.path.join(save_path, "r" + filename), result)
    print("complete")
    return


def create_set(path, save_path, size=500):
    """
    Take dataset, shuffle images and extract a subset

    @params:
        size: Size of a new set
    """
    files = os.listdir(path)
    random.shuffle(files)
    for filename in files:
        size = size - 1
        if size < 0:
            break
        copyfile(os.path.join(path, filename), os.path.join(save_path, "n" + str(size)) + ".jpg")
    return


def create_sets(percentage=0.9):
    """
    Creates test and training sets from all preprocessed images

    @params:
        percentage: Percentage of training set
    """
    # randomly divide images from training+test folder in test and training sets
    path = os.path.join(os.getcwd(), 'data\\training+test')
    path_test = os.path.join(os.getcwd(), 'data\\test')
    path_training = os.path.join(os.getcwd(), 'data\\training')
    files = os.listdir(path)
    random.shuffle(files)

    size = len(files)
    test_idx = train_idx = 0
    for filename in files:
        if train_idx < size * percentage:
            train_idx = train_idx + 1
            name = "train" + str(train_idx)
            copyfile(os.path.join(path, filename), os.path.join(path_training, name + ".jpg"))
        else:
            test_idx = test_idx + 1
            name = "test" + str(test_idx)
            copyfile(os.path.join(path, filename), os.path.join(path_test, name + ".jpg"))
    return


def normalize(path, save_path):
    """
    Normalize images, adjust pixel distribution
    """
    files = os.listdir(path)

    for filename in files:
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        result = cv2.normalize(img, None, 0, 235, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(save_path, "n" + filename), result)

    return

