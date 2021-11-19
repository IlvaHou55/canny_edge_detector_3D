# canny_edge_detector_3D
Perform Canny edge detection in 3D in python.

Based on the [step-by-step implementation of the Canny edge detection](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123) from @FienSoP, I derived a 3D version for application on 3D images. This is comparable with [edge3](https://nl.mathworks.com/help/images/ref/edge3.html) from the Image Processing Toolbox of Matlab. 

canny_edge_detector_3D.py includes the actual edge detection.
slicer_3D.py is a useful tool to scroll through 3D images for visualizing your results. Keep in mind that the edges go in three directions and only 2D images are displayed using the slicer.

The parameters like sigma, lowthresholdratio and highthresholdratio can be set according to preferences of your image and application.

Canny_Edge_Detection_3D_Example_Notebook.ipynb is an example notebook you can use and adjust for your application. For this specific example I made use of [this](https://figshare.com/s/2904b1ee61c3240f9291) openly available MRI dataset. I used 'MRI_4.nii' for this case.

The results from the Notebook look like this:

![image](https://user-images.githubusercontent.com/93598891/142627673-6425a0ac-304d-4c6a-aff5-d55b0c86d69f.png)

Hopefully this tool is useful for others as well and it will save time writing it!


