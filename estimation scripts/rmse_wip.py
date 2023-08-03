# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:47:35 2021

@author: Shawn
"""

from sklearn.metrics import mean_squared_error
from math import sqrt

rms_x = sqrt(mean_squared_error(x_actual, x_pt))
rms_y = sqrt(mean_squared_error(y_actual, y_pt))



