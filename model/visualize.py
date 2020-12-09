
import numpy as np
import cv2

example_file_path = "ABSOLUTE_FILE_PATH_HERE"

data = np.load(example_file_path)

ex_img = data[f"arr_{0}"]
ex_boxes = data[f"arr_{1}"]
ex_classes = data[f"arr_{2}"]

print("raw:")
print(ex_boxes, ex_classes)

