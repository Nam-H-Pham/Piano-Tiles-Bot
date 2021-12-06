import numpy as np
import PIL
from PIL import ImageGrab
import cv2
import Tkinter as tk
import copy
import matplotlib.pyplot as plt
import pyautogui
import time
import ctypes
import win32api
from mss import mss

raw_input("start:")

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

lower_range = np.array([0, 0, 0], dtype=np.uint8)
upper_range = np.array([50,50,100], dtype=np.uint8)

sct = mss()
monitor = {'top': 0, 'left': 0, 'width': screen_width, 'height': screen_height}

while True:
    img = np.array(sct.grab(monitor))
    #img = ImageGrab.grab(bbox=(0,0,screen_width,screen_height)) 
    img_np = np.array(img)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)


    #169, 100, 100 # red 0, 100, 100  #0, 0, 0       black
    #189, 255, 255       20, 255, 255 #180, 255, 50

    mask = cv2.inRange(hsv, lower_range, upper_range)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    
    try:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        x,y,w,h = cv2.boundingRect(biggest_contour)
        if w < 150 or h > 150:
            h = h/4
            y = y + 2*h
            #mask = img_np.copy()
            mask = img_np[y+30:y+h - 30,x + 10:x+w-10]
            #cv2.rectangle(img_np,(x,y),(x+w,y+h),(0,0,255),2)

            image = mask.copy()
            if True:
                i = 15#h/2
                check = []
                check.append(image.shape[1]/4 - 10)
                check.append((image.shape[1]/4)*2 - 10)
                check.append((image.shape[1]/4)*3 - 10)
                check.append(image.shape[1] - 10)
                for k in check:
                    if image[i,k,0] == 17:# and image[i,k,1] == 17 and image[i,k,2] == 17:
                        #cv2.rectangle(mask,(k,i),(k+1,i+1),(0,0,255),2)
                    
                        clickpos = (x + k, y + i)
                        pyautogui.PAUSE = 0.0
                        pyautogui.click(clickpos)
                        #cv2.circle(img_np,(clickpos), 5, (0,255,0), -1)
                        break
                        
            
    except:
        pass

    #cv2.imshow("screen",img_np)
    #cv2.imshow("mask",mask)


    if cv2.waitKey(20) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
