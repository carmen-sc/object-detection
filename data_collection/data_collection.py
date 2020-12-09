import numpy as np
import cv2

from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs("./data_collection/dataset"):
        np.savez(f"./data_collection/dataset/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1


def clean_segmented_image(seg_img):
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes

    # iterate over different possible classes
    for class_number in range(1, 5):
        # define mask bounds
        if class_number == 1:
            # Set range for "duckie color" (purple)
            color = np.array([100, 117, 226])
        elif class_number == 2:
            # Set range for "cone color" (pink)
            color = np.array([226, 111, 101])
        elif class_number == 3:
            # Set range for "truck color" (grey)
            color = np.array([116, 114, 117])
        elif class_number == 4:
            # Set range for "bus color" (yellow)
            color = np.array([216, 171, 15])
        else:
            print("somethings not right with mask bound definition")
            exit()

        # apply mask
        mask_bol = np.all(seg_img == color, axis=-1)
        mask = mask_bol.astype(np.uint8)

        # clean away snow
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # draw boxes
        for contour in contours:
            # cv2.rectangle(seg_img, (pack_box_array(cv2.boundingRect(contour))[0], pack_box_array(cv2.boundingRect(contour))[1]), (pack_box_array(cv2.boundingRect(contour))[2], pack_box_array(cv2.boundingRect(contour))[3]), (0, 255, 0), 2)
            try:
                boxes = np.append(boxes, [pack_box_array(cv2.boundingRect(contour))], axis=0)
                classes = np.append(classes, np.array(class_number))
            except Exception as e:
                # print(e)
                boxes = np.array([pack_box_array(cv2.boundingRect(contour))])
                classes = np.array([class_number])
        #display_seg_mask(seg_img, mask)

    return boxes.astype(int), classes


def pack_box_array(values):
    x, y, w, h = values
    packed_values = np.array([x, y, x+w, y+h])
    return packed_values


seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0
    kernel = np.ones((5, 5), np.uint8)

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)

        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        boxes, classes = clean_segmented_image(segmented_obs)
        save_npz(obs, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
