#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt

class Slicer3D:
    def __init__(self, ax, X):
        self.ax = ax
        self.name = [name for name in globals() if globals()[name] is X]
        ax.set_title(f'{self.name}')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        
        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_xlabel('slice %s' % self.ind)
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.im.axes.figure.canvas.draw()

