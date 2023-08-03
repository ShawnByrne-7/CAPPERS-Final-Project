# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:56:01 2021

@author: Shawn
"""
                    ####--- The CAPPERS Project ---####

#Import everything necessary

import sys
#sys.path.append('C:/Users/Libraries/Downloads/Cappers Use This')

import util as cm
import cv2
import os
from skeletontracker import skeletontracker
import pandas as pd
from datetime import datetime
import numpy as np
confidence_threshold = .5
import pyrealsense2 as rs
import numpy as np
from PIL import Image


#Need Bag file from Intel Realsense


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

bag_file_name = input("Enter a bag file name (Add .bag to the end). Hit ESC when you want the stream to end:")
#Frame_input = input ("Enter a frame:")

overlay_folder = input("Name the new folder which will store the output images: ")
os.mkdir(overlay_folder)

#Enable Streams for videos from bag file
rs.config.enable_device_from_file(config, bag_file_name)

#May need to change resolution and format.bgr depending on settings in RealSense viewer
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
colorizer = rs.colorizer();
skeletrack = skeletontracker()

#Create data frame with all labels
joint_labels_2D = ['NOSE_X','NOSE_Y','JUGULAR NOTCH_X','JUGULAR NOTCH_Y','R_SHOULDER_X','R_SHOULDER_Y',\
                'R_ELBOW_X','R_ELBOW_Y','R_WRIST_X','R_WRIST_Y','L_SHOULDER_X','L_SHOULDER_Y','L_ELBOW_X','L_ELBOW_Y',\
                'L_WRIST_X','L_WRIST_Y','R_HIP_X','R_HIP_Y','R_KNEE_X','R_KNEE_Y','R_ANKLE_X','R_ANKLE_Y',\
                'L_HIP_X','L_HIP_Y','L_KNEE_X','L_KNEE_Y','L_ANKLE_X','L_ANKLE_Y','R_EYE_X','R_EYE_Y',\
                'L_EYE_X','L_EYE_Y','R_EAR_X','R_EAR_Y','L_EAR_X','L_EAR_Y']
df = pd.DataFrame(columns=joint_labels_2D)
for i in range(len(joint_labels_2D),0, -2):
    yomama = joint_labels_2D[i-1]
    yomama = yomama[:-1] + 'D'
    df.insert(i, yomama, 69)


counter = 0
success= True
try:
    while success:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        
        depth_color_frame = colorizer.colorize(depth_frame)
        
        # Convert depth_frame to numpy array to render image in opencv
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        depth_frames = np.asanyarray(depth_frame.get_data())
        
        #Create/save the images
        
        #im1 = Image.open(r'Color.jpg')
        #im1.save(r'Color.png')
        
        #isSaved = cv2.imwrite("Depth.jpg{:d}.jpg".format(counter), depth_color_image)
        #isSaved = cv2.imwrite("Color{:d}.jpg".format(counter), color_image)
        
        #skeleton tracking
        #fix joint labels later if you can get for loops to work
        
        
        img = cv2.imread("Color{:d}.jpg".format(counter))
        
        #Resize resolution to match color and depth images
        res = cv2.resize(img, dsize=(1024, 768), interpolation=cv2.INTER_CUBIC)
        #run Skeleton Tracker
        skeletons = skeletrack.track_skeletons(res) 
        
        
        #Add the depth coordinate
        
        #np.set_printoptions(threshold=sys.maxsize)

        pd.set_option("display.max_rows", None, "display.max_columns", None)
        rows = []
        for pixel in range(768):
            rows.append("row"+"_"+str(pixel))
        df_1 = pd.DataFrame(depth_frames, rows)
        
        #if you want to see the entire images depth frames
        #df_1.to_excel("depth_frames.xlsx")
        
        frame_coords = [] # list to store coordinates for current stillframe
       
        for joint in range(0,18): # extract 18 joints tracked
            for xy in range(0,2): # extract x and y coords for each joint      
                frame_coords.append(skeletons[0][0][joint][xy])
            frame_coords.append(0)
        frame_coords = pd.Series(frame_coords, index = df.columns)
        df = df.append(frame_coords,ignore_index=True)   

        
        #Append dataframe with new found depth coordinate  
        for c in range(0, len(df.columns), 3):
    
            pixel_x = int(df.iloc[counter][c])
            pixel_y = int(df.iloc[counter][c+1])
            depth=df_1[pixel_x][pixel_y]
            df.iat[counter,c+2]=depth
            

        #Save the overlapped image to the folder 
        cm.render_result(skeletons, res, confidence_threshold)
        #img_out = "frame{:d}_skeleton.jpg".format(counter)
        cv2.imwrite(os.path.join(overlay_folder,"res{:d}.jpg".format(counter)), res) # Save the frame to raw folder
        
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        
        counter +=1
        if key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pass

#startTime = datetime.now()
#myvideo = cv2.VideoCapture(filename)
#success,image = myvideo.read()



    
#print(datetime.now() - startTime) # displays how long it took to process the video
#print("The result images are saved in folder: ", overlay_folder)
df.to_csv(str(overlay_folder+'.csv'), index=False)
print("The csv containing the joint data is labeled: {:s}.csv".format(overlay_folder))  