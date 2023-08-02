from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import cv2 as cv
import math
import numpy as np
from tkinter.ttk import *

root = Tk()
root.geometry("512x512")
root.resizable(width=True, height=True)

s = Style(root)
# add the label to the progressbar style
s.layout("LabeledProgressbar",
         [('LabeledProgressbar.trough',
           {'children': [('LabeledProgressbar.pbar',
                          {'side': 'left', 'sticky': 'ns'}),
                         ("LabeledProgressbar.label",   # label inside the bar
                          {"sticky": ""})],
           'sticky': 'nswe'})])

def img_matching_crop():
    global kp1, des1, sift, template, save_path, similaliry_score
    for image_full_path in range(len(all_file_name)):
        full_path = all_file_name[image_full_path]
        img_to_crop_bgr = cv.imread(cv.samples.findFile(full_path))
        img_to_crop_gray = cv.cvtColor(img_to_crop_bgr, cv.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(img_to_crop_gray,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        matchesMask = [[0,0] for i in range(len(matches))]

        good = []
        point_x = 0
        point_y = 0
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                matchesMask[i]=[1,0]
                point_x = math.ceil(kp2[m.trainIdx].pt[0]) - math.ceil(kp1[m.queryIdx].pt[0])
                point_y = math.ceil(kp2[m.trainIdx].pt[1]) - math.ceil(kp1[m.queryIdx].pt[1])
                if point_x>0 and point_y>0 and (point_x+template.shape[1])<img_to_crop_gray.shape[1] and (point_y+template.shape[0])<img_to_crop_gray.shape[0]:
                    good.append([math.ceil(point_x),math.ceil(point_y)])
        good = sorted(good, key=lambda x: x[0])
        all_point = []
        for i in range(len(good)-1):
            if abs(good[i+1][0] - good[i][0]) > 100 or abs(good[i+1][1] - good[i][1]) > 100:
                all_point.append(good[i+1])
                if i == 0:
                    all_point.append(good[i])
        for index, (point_x, point_y) in enumerate(all_point):
            res = cv.matchTemplate(img_to_crop_gray[int(point_y):int(point_y)+template.shape[0], int(point_x):int(point_x)+template.shape[1]],template,cv.TM_CCORR_NORMED)
            img_save = img_to_crop_bgr[int(point_y):int(point_y)+template.shape[0], int(point_x):int(point_x)+template.shape[1], :]
            if np.squeeze(res) > float(similaliry_score.get()):
                filename = full_path.split("\\")[-1].split(".")
                filename = save_path + "\\" + "".join(filename[:-1]) + "_" +str(index) + "." + filename[-1]
                cv.imwrite(filename, img_save)
        progress['value'] += (1/len(all_file_name))*100
        s.configure("LabeledProgressbar", text="{0} %      ".format(progress['value']))
        root.update_idletasks()
    progress['value'] = 0

def list_all_file():
    global all_file_name
    all_file_name = []
    path_all = opendir()
    for path, subdirs, files in os.walk(path_all):
        for name in files:
            if "png" in name or "bmp" in name:
                all_file_name.append(os.path.join(path, name))

def save_path_fn():
    global save_path
    save_path = opendir()
    

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def opendir():
    filename = filedialog.askdirectory(title='open')
    return filename

def open_img():
    global template, sift, kp1, des1, progress
    x = openfn()
    img1 = cv.imread(cv.samples.findFile(x), cv.IMREAD_GRAYSCALE)
    cv.namedWindow("select the area", cv.WINDOW_NORMAL)
    cv.resizeWindow("select the area", img1.shape[1]//3 , img1.shape[0]//3)
    r = cv.selectROI("select the area", img1)
    cv.destroyWindow("select the area")
    img1 = img1[int(r[1]):int(r[1]+r[3]), 
                        int(r[0]):int(r[0]+r[2])]
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    template = img1
    img = ImageTk.PhotoImage(Image.fromarray(img1))
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
    progress = Progressbar(root, orient = HORIZONTAL,
              length = 300, style="LabeledProgressbar")
    progress.pack(pady = 10)


btn = Button(root, text='Template', command=open_img).pack()
part_matching = Button(root, text='Part matching', command=list_all_file).pack()
get_path_save = Button(root, text='Save path', command=save_path_fn).pack()
Label(root, text="Enter Matching Score", font=('Calibri 10')).pack()
similaliry_score=Entry(root, width=35)
similaliry_score.pack()
start_matching = Button(root, text='Start', command=img_matching_crop).pack(pady = 10)


root.mainloop()