from model import Wrapper, Model
import numpy as np
import cv2

model = Model()
wrapper = Wrapper()

example_file_path = "/media/carmen/Windows/Linux_File_Stash/object-detection-ex-template/eval/dataset/5.npz"

data = np.load(example_file_path)

ex_img = data[f"arr_{0}"]
ex_boxes = data[f"arr_{1}"]
ex_classes = data[f"arr_{2}"]

predicted_boxes, predicted_labels, predicted_scores = wrapper.predict(ex_img)

print("raw:")
print(ex_boxes, ex_classes)
print("model:")
print(predicted_boxes, predicted_labels, predicted_scores)

for entry in predicted_boxes:
    for box in entry:
        point_low = (box[0], box[1])
        point_high = (box[2], box[3])
        ex_img = cv2.rectangle(ex_img, point_low, point_high, (0, 255, 0), 2)

cv2.imshow("image", ex_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
