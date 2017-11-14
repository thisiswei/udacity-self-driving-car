# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.
1. convert the images to grayscale
2. blur the images using gaussian blur
3. find the edges using canny
4. mask out the edges, and finally
5. convert the image to hough space and connect the lines

In order to draw a single line on the left and right lanes, I modified the draw_lines() function:

1. separates lines into two groups -- left land lines and right lane lines -- left lane lines's slope are >0 and right lane lines <0

2. for each group: calculate the average point and the average slope 

3. for each group: extrapolate the lines by combinations of the following:
   - find the top point (TP) and the bottom point (BP), and
   - connect TP and BP
   - the TP and BP are calculated using:
     - m*x + b = y, and
     - TP's y is 310 (the y of the mask defined in piple step 4), and combine with the fomular above we get TP's x
     - using same logic we get BP



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

the slopes are the same for the both lanes, (e.g., left turn or right turn) and my solution will probably draw only one line



### 3. Suggest possible improvements to your pipeline

A possible improvement would be group the lines by their relatives positions and left lane lines will always be separated from the right lane lines


