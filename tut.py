# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pybullet as p

p.connect(p.GUI)

p.loadURDF("plane.urdf",0,0,-1)
quadcopterId = p.loadURDF("quadrotor.urdf")