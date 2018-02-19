# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:36:06 2017

@author: Mitchel
"""

import numpy as np
import matplotlib.pylab as plt
import control

teller = np.array([1.0])
noemer = np.array([1.0, -1.0])

K = 0.1
H = control.tf(teller,noemer)
print(H)

Sys1 = K*H
Sys2 = 1
Hclosed = control.feedback(Sys1,Sys2)
print(Hclosed)