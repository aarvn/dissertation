import os
import xml.etree.ElementTree as ET
import json
import numpy as np

# Sources:
# polyArea - https://stackoverflow.com/a/30408825


def onehot(val, count):
    """Generate one hot encoded value."""
    ret = [0]*count
    ret[val] = 1
    return ret


# Initialise category map
categorymap = {
    "fashion": 0,
    "food": 1,
    "news": 2,
    "science": 3,
    "travel": 4,
    "wedding": 5
}


def getLabels(path, file_name):
    """Calculate the label for a layout annotation."""
    def polyArea(x, y):
        """Calculate area of polygon."""
        """Source: https://stackoverflow.com/a/30408825"""
        return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

    # Initialise area classes
    category = file_name.split("_")[0]
    textArea = 0
    imageArea = 0
    headline = 0

    # Parse xml file to calculate image area, text area and headline count
    root = ET.parse(path).getroot()
    for layout in root.findall('layout'):
        for element in layout.findall('element'):
            label = element.get('label')

            px = [int(float(i)) for i in element.get(
                'polygon_x').split(" ") if i != "NaN"]
            py = [int(float(i)) for i in element.get(
                'polygon_y').split(" ") if i != "NaN"]

            area = polyArea(px, py)

            if(label == "image"):
                imageArea += area
            elif (label in ["text", "text-over-image"]):
                textArea += area
            elif (label == "headline" or label == "headline-over-image"):
                headline = 1

    # Assign category to label
    category = categorymap[category]

    # Assign text proportion to label
    textProportion = textArea/(300*225)
    textProportion = round(textProportion, 1)
    textProportion = max(0, min(textProportion, 0.7))
    textProportion = int(textProportion * 10)

    # Assign image proportion to label
    imageProportion = imageArea/(300*225)
    imageProportion = round(imageProportion, 1)
    imageProportion = max(0, min(imageProportion, 1))
    imageProportion = int(imageProportion * 10)

    return {
        "category": onehot(category, len(categorymap)),
        "textProportion": onehot(textProportion, 8),
        "imageProportion": onehot(imageProportion, 11),
        "headlines": onehot(headline, 2)
    }


# Generate labels
annotations = os.listdir("annotations")
for annotation in annotations:
    label = getLabels('annotations/'+annotation, annotation)
    with open("layout_labels/" + annotation.split(".")[0]+".json", 'w') as f:
        json.dump(label, f)
