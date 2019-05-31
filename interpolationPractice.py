#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:20:19 2019

@author: watkins35
code from https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
"""
import matplotlib.pyplot as py
from scipy.interpolate import interp1d

x = np.linspace(0, 10,11)
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')


xnew = np.linspace(0, 10, num=41, endpoint=True)

py.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
py.legend(['data', 'linear', 'cubic'], loc='best')
py.show()
