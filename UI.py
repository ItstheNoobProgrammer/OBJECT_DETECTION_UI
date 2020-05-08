# -*- coding: utf-8 -*-
"""*******************************************************************************************************
File Name   : main.py
Author      : Mukund R
Date        : 05/05/2020    
Description : UI for object detection mainly to choose a video file and show predicton in cv window  
*******************************************************************************************************"""
#Tk is a free and open-source, cross-platform widget toolkit that provides a library of basic elements 
#of GUI widgets for building a graphical user interface in many programming languages.
#Tkinter is a Python binding to the Tk GUI toolkit.
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image 
import cv2
import numpy as np
from darkflow.net.build import TFNet 


global img
global panel

"""*******************************************************************************************************
Method Name : browseFiles
Author      : Mukund R
Date        : 05/05/2020    
Description : Browse the file from the filedialog and populate the var filename    
*******************************************************************************************************"""
def browseFiles():
    root.filename =  filedialog.askopenfilename(initialdir = "/",
                                                title = "Select file",
                                                filetypes = (("mp4 files","*.mp4"),("all files","*.*")))

    #Logic for thumbnail
    #read the video
    vidcap = cv2.VideoCapture(root.filename)
    success,image = vidcap.read()
    #TODO Create DIR to store all these
    #strip 1st frame and write it to a file
    cv2.imwrite("C:/Users/mukun/Documents/soccer_ball_data/tumbnail/tumbnail.jpg", image)

    #repaint the frame with opened image
    # opens the image 
    img = Image.open("C:/Users/mukun/Documents/soccer_ball_data/tumbnail/tumbnail.jpg") 
	
    # resize the image and apply a high-quality down sampling filter 
    img = img.resize((512, 512), Image.ANTIALIAS) 

    # PhotoImage class is used to add image to widgets, icons etc 
    img = ImageTk.PhotoImage(img) 

    # create a label 
    panel = tk.Label(root, image = img) 
	
    # set the image as img 
    panel.image = img 
    panel.grid(column = 1,row = 1,padx = 250, pady = 50) 

"""*******************************************************************************************************
method Name : boxing
Author      : Mukund R
Date        : 05/05/2020    
Description : # using Opencv draw a bounding box rectangle with confidance value as label      
*******************************************************************************************************"""    
def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage
"""*******************************************************************************************************
method Name : on_button_next
Author      : Mukund R
Date        : 05/05/2020    
Description : # Do the prediction and render it in opencv frame      
*******************************************************************************************************"""    
# function to open a new window 
# on a button click 
def on_button_next(): 
    
    options = {"model": "C:/Users/mukun/Documents/darkflow/cfg/yolov2.cfg",
               "load": -1,
               "gpu": 1.0}
    tfnet2 = TFNet(options)
    tfnet2.load_from_ckpt()
    print(root.filename)
    cap = cv2.VideoCapture(root.filename)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #TODO Create DIR to store all these
    out = cv2.VideoWriter('C:/Users/mukun/Documents/soccer_ball_data/sample_video/output1.mp4',fourcc, 20.0, (int(width), int(height)))
   
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if ret == True:
            frame = np.asarray(frame)      
            results = tfnet2.return_predict(frame)
        
            new_frame = boxing(frame, results)

            # Display the resulting frame
            out.write(new_frame)
            cv2.imshow('frame', new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
              
"""*******************************************************************************************************
Class Name : Application
Author      : Mukund R
Date        : 05/05/2020    
Description : Window1:
                Widgets Included: Top label, File Browser, Image viwer, Next button and exit button  
                # Grid method is chosen for placing the widgets at respective positions  
                # in a table like structure by specifying rows and columns         
*******************************************************************************************************"""
class Application(tk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        #Top Label
        tk.Label(root, 
                 text="Tiger Detection",
                 fg = "orange",
                 bg= "dark slate gray",
                 font = "Helvetica 64 bold italic").grid(column = 1, row = 0,padx = 250, pady = 50)
        
               
        tk.Button(root,  
                  text = "Browse Files", 
                  command = browseFiles,
                  fg = "orange",
                  bg = "dark slate gray",
                  font = "Helvetica 30 bold").grid(column = 0, row = 2,padx=50, pady=50)   
         
        #default frame
        # opens the image 
        img = Image.open("images.jpg") 
	
        # resize the image and apply a high-quality down sampling filter 
        img = img.resize((512, 512), Image.ANTIALIAS) 

        # PhotoImage class is used to add image to widgets, icons etc 
        img = ImageTk.PhotoImage(img) 

        # create a label 
        panel = tk.Label(root, image = img) 
	
        # set the image as img 
        panel.image = img 
        panel.grid(column = 1,row = 1,padx = 250, pady = 50)  
        
        #Exit button
        tk.Button(root, text="Next", fg = "orange",
                  bg = "dark slate gray",font = "Helvetica 16",
                              command=on_button_next).grid(column = 2, row = 3,padx = 5,pady=5)
        #Exit button
        tk.Button(root, text="QUIT", fg = "orange",
                  bg = "dark slate gray",font = "Helvetica 16",
                              command=self.master.destroy).grid(column = 3, row = 3, pady=5)
   
"""*******************************************************************************************************
Method Name : N/A
Author      : Mukund R
Date        : 05/05/2020    
Description : Driver code Init TK and start building GUI by calling Method "Application" 
*******************************************************************************************************"""
# Create the root window 
root = tk.Tk()
root.attributes('-fullscreen', True)
#Set window background color 
root.config(background = "dark slate gray") 
#init the var file browser to store the src img 
root.filename = ''
#invoke class application
app = Application(master=root)
app.mainloop()   
