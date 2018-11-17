from zipfile import ZipFile
import glob
import os


def extractor(zipFilePath):

    # Loading the Zipfile
    fileHeader = ZipFile(zipFilePath)
    print('ZipFile found')

    # Check & unzip the Zipfile
    print('Checking if file extracted...')
    if os.path.isdir('dog-data') == 0:
        print('Extracting.....')
        fileHeader.extractall()
    else:
        print('File already extracted.')

    # Getting the file path to images in both testing & training datasets
    trainPath = glob.glob('Dog-data/dog-training/*.tif')
    testPath = glob.glob('Dog-data/dog-test/*.tif')
    print('Image path obtained')

    # Initiating Array for class labels
    trainLabel = []
    testLabel = []

    # Assigning training Class labels
    for label in trainPath:
        if label.find('cat.') > 0:
            trainLabel.append(0)
        if label.find('dog.') > 0:
            trainLabel.append(1)

    # Assigning testing Class labels
    for label in testPath:
        if label.find('cat.') > 0:
            testLabel.append(0)
        if label.find('dog.') > 0:
            testLabel.append(1)

    print('Class labels obtained')
    print('Data Extraction Done!!')

    return trainPath, trainLabel, testPath, testLabel