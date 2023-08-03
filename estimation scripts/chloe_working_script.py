# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:19:41 2021

@author: Chloe Keller
"""

import sys
sys.path.append('C:/Users/Libraries/Downloads/Cappers Use This')
import util as cm
import cv2
import os
from skeletontracker import skeletontracker
import pandas as pd
from datetime import datetime

confidence_threshold = .5

# Format dataframe to store coordinates
joint_labels = ['NOSE_X','NOSE_Y','JUGULAR NOTCH_X','JUGULAR NOTCH_Y','R_SHOULDER_X','R_SHOULDER_Y',\
                'R_ELBOW_X','R_ELBOW_Y','R_WRIST_X','R_WRIST_Y','L_SHOULDER_X','L_SHOULDER_Y','L_ELBOW_X','L_ELBOW_Y',\
                'L_WRIST_X','L_WRIST_Y','R_HIP_X','R_HIP_Y','R_KNEE_X','R_KNEE_Y','R_ANKLE_X','R_ANKLE_Y',\
                'L_HIP_X','L_HIP_Y','L_KNEE_X','L_KNEE_Y','L_ANKLE_X','L_ANKLE_Y','R_EYE_X','R_EYE_Y',\
                'L_EYE_X','L_EYE_Y','R_EAR_X','R_EAR_Y','L_EAR_X','L_EAR_Y']
df = pd.DataFrame(columns=joint_labels)


# Input Video Name
#filename = "side_step.mov"
filename = input("Enter video file name (be sure to include extension .mov .mp4 etc): ")

# Name folder to store all extracted frames (don't need if we only want skel imgs)
#raw_folder = 'split_test'
#os.mkdir(raw_folder)

# Name folder to store skeleton overlays
overlay_folder = input("Name the new folder which will store the output images: ")
os.mkdir(overlay_folder)

startTime = datetime.now()

skeletrack = skeletontracker() # Create new skeleton tracking object
myvideo = cv2.VideoCapture(filename)
success,image = myvideo.read()
count = 0
success=True

while success: 

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) # for some reason it keeps rotating the video
    print("Read New Frame ", count, "//", success)
    #cv2.imwrite(os.path.join(raw_folder,"frame{:d}.jpg".format(count)), image) # Save the frame to raw folder

    skeletons = skeletrack.track_skeletons(image)  # Perform skeleton tracking
    
    # Extract x,y coordinates from skeleton
    frame_coords = [] # list to store coordinates for current stillframe
    for joint in range(0,18): # extract 18 joints tracked
        for xy in range(0,2): # extract x and y coords for each joint
            frame_coords.append(skeletons[0][0][joint][xy])
    frame_coords = pd.Series(frame_coords, index = df.columns)
    df = df.append(frame_coords,ignore_index=True) # add current frame data to df    
    
    
    # Overlay the skeleton on the original image and save to folder
    cm.render_result(skeletons, image, confidence_threshold)
    img_out = "frame{:d}_skeleton.jpg".format(count)
        
    if img_out:
        isSaved = cv2.imwrite(os.path.join(overlay_folder,img_out), image)        
        if isSaved:
            print("")
        else:
            print("Saving the result image failed for the given path: ", img_out) #If things go wrong, this will help   


    count+=1
    success,image = myvideo.read()
    
#print(datetime.now() - startTime) # displays how long it took to process the video
print("The result images are saved in folder: ", overlay_folder)
df.to_csv(str(overlay_folder+'.csv'), index=False)
print("The csv containing the joint data is labeled: {:s}.csv".format(overlay_folder))
