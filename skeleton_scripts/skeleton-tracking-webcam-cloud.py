#!/usr/bin/env python3
from skeletontracker import skeletontracker
import util as cm
import cv2

# entry point
if __name__ == "__main__":
    try:
        cloud_tracking_api_key = cm.get_cloud_tracking_api_key()

        # initialise the webcam
        capture_device = cv2.VideoCapture(0) 
        hasFrame, frame = capture_device.read()

        # initialize the cubemos skeleton tracking 
        skeletrack = skeletontracker(cloud_tracking_api_key)
        joint_confidence = 0.2

        # create window for initialisation
        window_name = "cubemos skeleton tracking with webcam as input source"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)     
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while hasFrame:
            # perform inference and update the tracking id
            skeletons = skeletrack.track_skeletons(frame)

            # render the skeletons on top of the acquired image and display it
            cm.render_result(skeletons, frame, joint_confidence)
            cm.render_ids(skeletons, frame)
            
            # show the result on on opencv window
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) == 27:
                break;

            # Capture a new frame
            hasFrame, frame = capture_device.read()

    except Exception as ex:
        print("Exception occured: \"{}\"".format(ex))