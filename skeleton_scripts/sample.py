# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:12:23 2021

@author: Shawn
"""

#!/usr/bin/env python3
import util as cm
import cv2
import argparse
import os
import platform
from skeletontracker import skeletontracker
'''
parser = argparse.ArgumentParser(description="Perform keypoing estimation on an image")
parser.add_argument(
    "-c",
    "--confidence_threshold",
    type=float,
    default=0.5,
    help="Minimum confidence (0-1) of displayed joints",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Increase output verbosity by enabling backend logging",
)

parser.add_argument(
    "-o",
    "--output_image",
    type=str,
    help="filename of the output image",
)

parser.add_argument("image", metavar="I", type=str, help="filename of the input image")
'''

# Main content begins
if __name__ == "__main__":
    try:
        # Parse command line arguments and check the essentials
        #args = parser.parse_args()
        img_path = "skeleton_estimation.jpg"
        #img_path = "C:\\Users\\Shawn\\Cubemos-Samples\\cubemos-SkeletonTracking_3.1.0.745b3b5\\data.tar\\data\\opt\\cubemos\\skeleton_tracking\\samples\\res\\images\\skeleton_estimation.jpg" 
        # Read the image
        # img = cv2.imread(args.image)
        img_out = "Out_chloe.jpg"
        #img_out = "C:\\Users\\Shawn\\Desktop\\Out.jpg"
        confidence_threshold = .5
        img = cv2.imread(img_path)
        # Get the skeleton tracking object
        skeletrack = skeletontracker()

        # Perform skeleton tracking
        skeletons = skeletrack.track_skeletons(img)

        # Render results
        cm.render_result(skeletons, img, confidence_threshold)
        print("Detected skeletons: ", len(skeletons))
        #if args.verbose:
        print(skeletons)
        f= open("Skeleton_chloe.txt",'w')  
        f.write(str(skeletons))
        f.close()
        
        if img_out:
            isSaved = cv2.imwrite(img_out, img)
            if isSaved:
                print("The result image is saved in: ", img_out)
            else:
                print("Saving the result image failed for the given path: ", img_out)


            
    except Exception as ex:
        print("Exception occured: \"{}\"".format(ex))
# Main content ends
