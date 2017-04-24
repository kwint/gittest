# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:15:55 2017

@author: Mitchel
"""

import cv2
import numpy as np
import time
import os
#from skimage.measure import compare_ssim as ssim
nummers = os.listdir("./Nummerbord/Template Jelle")
def nothing(x):
    pass

def invertimg(img):
    img = abs(255-img)
    return(img)
def cropandwarp (img, rect, w, h, yplus, ymin):
    W = rect[1][0] - w
    H = rect[1][1] - h
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys) + ymin
    y2 = max(Ys) + yplus
    
    angle = rect[2]
    if angle < -45:
        angle += 90
        
    # Center of rectangle in source image
    center = ((x1+x2)/2,(y1+y2)/2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
    return croppedRotated


cv2.namedWindow('frame',cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('frame',800,0)

cv2.namedWindow('res',cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('res',0,0)

cv2.namedWindow('bin',cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('bin',600,600)

while (1):
    kentekenfinal=[]
    threshold = 1.00
    # WERKT:
    
    # IMG_0985
    # NL06JFKF
    # NL06TFFZ
    # NL08KTD6
    # NL64SZTR
    # NL33RXR5
    # NL83LHL9
    
    #inlezen van het bestand
    name = input("welk bestand? ")
    InputPicture = cv2.imread('./nummerbord/%s' %name)
    InputPicture = cv2.resize(InputPicture,(600, 300),interpolation = cv2.INTER_CUBIC)
    #Omzetten naar HSV
    HsvPicture = cv2.cvtColor(InputPicture, cv2.COLOR_BGR2HSV)
    
    #Masker threshold van geel
    lower_yellow = np.array([10,100,170])
    upper_yellow = np.array([30,255,255])
    mask_yellow = cv2.inRange(HsvPicture, lower_yellow, upper_yellow)
    
    
    #Laat alleen geel zien (Het nummerbord)
    Result = cv2.bitwise_and(InputPicture,InputPicture, mask= mask_yellow)
    Resultgray = cv2.cvtColor(Result, cv2.COLOR_HSV2BGR)
    Resultgray = cv2.cvtColor(Resultgray, cv2.COLOR_BGR2GRAY)
    
    #Omzetten naar een zwart wit foto om makkelijker contouren te vinden
    _, binary = cv2.threshold(Resultgray, 127, 255, cv2.THRESH_BINARY)
    binary, contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    #Contours[0] is het contour met het grootste contourArea door de sort hierboven
    cnt = contours[0]
    #Teken een rechthoek en een huls om de witte delen van het masker
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box=np.int0(box)
    #Nummerbord bijsnijden en roteren
    croppedRotated = cropandwarp(InputPicture, rect, 6, 10, 1, 2)
    #Laten zien wat het programma geroteerd heeft op het originele plaatje
    cv2.drawContours(InputPicture, [box], -1, (255, 200, 200), 2)
    #Standaard maat maken voor de templates later
    nummerbord = cv2.resize(croppedRotated,(200, 40),interpolation = cv2.INTER_CUBIC)
    Sharpening_kernel = np.array([[-1,-1,-1,-1,-1],
                                  [-1,2,2,2,-1],
                                  [-1,2,8,2,-1],
                                  [-1,2,2,2,-1],
                                  [-1,-1,-1,-1,-1]]) / 8.0
    # applying different kernels to the input image
    nummerbord = cv2.filter2D(nummerbord, -1, Sharpening_kernel)
    
    
    
    
    ### hier hebben we alleen nog het nummerbord.
    nummerbord = cv2.cvtColor(nummerbord, cv2.COLOR_BGR2GRAY)
    _, nummerbordZwartWit = cv2.threshold(nummerbord, 127, 255, cv2.THRESH_BINARY)
    nummerbordWitZwart = invertimg(nummerbordZwartWit)
    #Eerst eroden en 2x dilaten om zoveel mogelijk ruis uit het plaatje te halen.
    #Daarna eroden om de karakters weer normaal te krijgen
    Dilatekernel = np.ones((2,2), np.uint8)
    Dilated = cv2.erode(nummerbordZwartWit,Dilatekernel,iterations = 1)
    Dilated = cv2.dilate(Dilated,Dilatekernel,iterations = 2)
    Dilated = cv2.erode(Dilated,Dilatekernel,iterations = 1)
    #Dilated, contours, hierachy = cv2.findContours(Dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    #Een kenteken heeft altijd 6 karakters
    while(len(kentekenfinal) < 6):
        #threshold begint op 100% match op de template.
        #als hij geen 6 overeenkomsten vindt, verlaagd hij de threshold en probeert het opnieuw
        threshold = threshold - 0.005
        #Kenteken arrays initialiseren
        kenteken = []
        kentekenfinal = []
        #Ga alle templates langs
        for file in nummers:        
            pos = -100
            #lees de template in
            comparingnumber = cv2.imread('./Nummerbord/Template Jelle/%s' %file)
            
            #De template resizen zodat deze in het plaatje past.
            width, height = comparingnumber.shape[::-2]
            Ratio = height/35
            Newwidth = width * Ratio
            comparingnumber = cv2.resize(comparingnumber,(25, 35),interpolation = cv2.INTER_CUBIC)
            comparingnumber = cv2.cvtColor(comparingnumber, cv2.COLOR_BGR2GRAY)
            
            w, h = comparingnumber.shape[::-1] 
            res = cv2.matchTemplate(Dilated,comparingnumber,cv2.TM_CCOEFF_NORMED)
            #Geef de locaties van elke plek waar de template overeen komt
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                #pt[0] is de positie linksonderin
                #De gevonden templates moeten minimaal 20 pixels van elkaar vandaan staan.
                #Om te voorkomen dat het programma dezelfde letter meerdere keren ziet
                   if (abs(pt[0]-pos) > 20 ):
                       pos = pt[0]
                       cv2.rectangle(nummerbord, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                       #Append karakter en het X coordinaat
                       kenteken.append([file[0], pos])
                       count = count + 1
        #Letters sorteren op het X coordinaat
        kenteken = sorted(kenteken, key=lambda kenteken_entry: kenteken_entry[1])
        
        prev=-20
        #Filteren of het programma niet bijvoorbeeld een 8 en B over elkaar heen wil zetten
        #Wederom door te kijken of de letters niet te dicht bij elkaar staan
        #Beste matches staan altijd iets meer naar links dan minder goede matches
        for i in range (0, len(kenteken)):
            if (abs(kenteken[i][1] - prev) > 10) :
                prev = kenteken[i][1]
                #Van een lijst met een karakters en coordinaat, naar een lijst met alleen karakters
                kentekenfinal.append(kenteken[i][0])
    print (kentekenfinal)    
    cv2.imshow('frame', Dilated)
    cv2.imshow('res', InputPicture)
    cv2.imshow('bin', nummerbord)
    
    #Druk op escape om het programma te sluiten
    if(cv2.waitKey(10)==27):
        break
    time.sleep(0.1)
cv2.destroyAllWindows() 