
"""
    Utility script to download cifar10 dataset and store it into the data/ directory.
"""
import os
import tarfile
import urllib.request

if __name__ == "__main__":
    CIFAR10_LINK = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    TEMP_NAME = "data/temp.tar.gz"
    urllib.request.urlretrieve (CIFAR10_LINK, TEMP_NAME)

    # unpack archiv
    tar = tarfile.open(TEMP_NAME, "r:gz")
    tar.extractall()

    # move the archive to the data folder
    os.rename("cifar-10-batches-bin", "data/cifar10-bin")

    # remove temporary file
    os.remove(TEMP_NAME)
