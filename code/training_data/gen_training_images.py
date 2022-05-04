# Convert annotations into images
import xml.etree.ElementTree as ET
import os
import math
import cv2 as cv
import numpy as np

# region Initialisations
# Color map
labelmap = {
    "background": (0, 0, 0),
    "image": (1, 0, 0),
    "headline": (0, 1, 0),
    "headline-over-image": (0, 1, 0),
    "text": (0, 0, 1),
    "text-over-image": (0, 0, 1),
}

imageHeight = 300
imageWidth = 225

encodedImageHeight = 32
encodedImageWidth = 24

dim = 32
xPad = math.floor((dim-encodedImageWidth)/2)
yPad = math.floor((dim-encodedImageHeight)/2)
# endregion


def encodeImage(path):
    """Create an image from an annotation file path."""
    def layoutToImage(path):
        """Create a large image from an annotation file path."""
        # create blank image
        image = np.zeros((imageHeight, imageWidth, 3), np.uint8)

        # draw polygons for different design elements
        root = ET.parse(path).getroot()
        for layout in root.findall("layout"):
            for element in layout.findall("element"):
                label = element.get("label")
                px = [int(float(i)) for i in element.get(
                    "polygon_x").split(" ") if i != "NaN"]
                py = [int(float(i)) for i in element.get(
                    "polygon_y").split(" ") if i != "NaN"]

                poly = np.array([list(zip(px, py))], dtype=np.int32)
                cv.fillPoly(image, poly, color=tuple(
                    [255*x for x in labelmap[label]]))

        return image

    # create a large image from XML file
    image = layoutToImage(path)

    # calculate strides (i.e. the width and height of each segment)
    yStride = math.floor(imageHeight/encodedImageHeight)
    xStride = math.floor(imageWidth/encodedImageWidth)

    # region encode/downsample image
    encodedImage = np.zeros((dim, dim, 3), np.uint8)
    for y in range(encodedImageHeight-1):
        for x in range(encodedImageWidth-1):
            # find region of interest
            col = yStride*y
            row = xStride*x
            img = image[col:col+yStride, row:row+xStride, :]

            # calculate most frequent color and assign to the encodedImage
            unique, counts = np.unique(
                img.reshape(-1, 3), axis=0, return_counts=True)
            encodedImage[yPad+y, xPad+x] = unique[np.argmax(counts)]
    # endregion

    return encodedImage


# Create training images
annotations = os.listdir("annotations")
for annotation in annotations:
    image = encodeImage("annotations/"+annotation)
    cv.imwrite("layout_images/" +
               annotation.split(".")[0]+".png", image)
