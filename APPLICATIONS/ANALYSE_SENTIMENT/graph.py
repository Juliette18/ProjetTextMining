# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:12:48 2019

@author: Juliette
"""

import matplotlib.pyplot as pp
import matplotlib.animation as ani
from matplotlib import style
import time

style.use("ggplot")
fig = pp.figure()
ax1 =fig.add_subplot(1,1,1)

def animation(i):
    pullData =open("twitter-out.txt","r").read()
    lines=pullData.split('\n')
    
    xar=[]
    yar=[]
    
    x=0
    y=0
    
    for l in lines:
        x+=1
        if "pos" in l:
            y+=1
        elif "neg" in l:
            y-=1
            
        xar.append(x)
        yar.append(y)
        
    ax1.clear()
    ax1.plot(xar,yar)

anim = ani.FuncAnimation(fig,animation,interval=1000)
pp.show()