import math
import os


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

_Y_HEIGHT = 320

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1))

    to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    left, right = _split_lines(lines)
    y_top = _Y_HEIGHT
    y_bottom = img.shape[1]

    if left:
        left_line = _get_line(left, y_top, y_bottom)
        cv2.line(img,

            (
                int(left_line['x_bottom']),
                int(left_line['y_bottom']),
            ),

            (
                int(left_line['x_top']),
                int(left_line['y_top']),
            ),

            color,
            thickness=thickness)

    if right:
        right_line = _get_line(right, y_top, y_bottom)
        cv2.line(
            img,

            (
                int(right_line['x_bottom']),
                int(right_line['y_bottom']),
            ),

            (
                int(right_line['x_top']),
                int(right_line['y_top']),
            ),

            color,
            thickness=thickness)


def _get_line(lines, y_top, y_bottom):
    avg_point = _get_avg_point(lines)
    avg_slope = _get_avg_slope(lines)

    b = _calculate_b(avg_point, avg_slope)
    x_top = _get_x(b, y_top, avg_slope)

    x_bottom = _get_x(b, y_bottom, avg_slope)
    return {
        'x_bottom': int(x_bottom),
        'y_bottom': int(y_bottom),
        'x_top': int(x_top),
        'y_top': int(y_top),
    }


def _get_avg_point(lines):
    '''
    Given lines return the avg point: (x,y)
    '''
    x_sum = sum(
        (x1+x2)/2.
        for line in lines
        for x1,_,x2,_ in line
    )
    y_sum = sum(
        (y1+y2)/2.
        for line in lines
        for _,y1,_,y2 in line
    )
    num_of_lines = len(lines) + 0.0

    return x_sum/num_of_lines, y_sum/num_of_lines


def _get_avg_slope(lines):
    '''
    Give lines return the average slope of the lines
    '''
    slope_sum = sum(
        ((y2-y1)/(x2-x1)+0.0)

        for line in lines
        for x1,y1,x2,y2 in line
    )
    return slope_sum / (len(lines) + 0.0)



def _get_x(b, y, slope):
    x = (y - b) / slope
    return x


def _calculate_b(point, slope):
    """
    calculate the b in the following formula:
    mx+b = y
    """
    x1, y1 = point
    # avg_right_slope * x1 + b = y1
    b = y1 - (slope * x1)
    return b


def _split_lines(lines):
    left = []
    right = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            if (slope > 0.45 and slope < 0.75):
                left.append(line)
            elif (slope < -0.6 and slope > -0.9):
                right.append(line)

    return left, right



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    '''
    it says, "try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines."
    but I can't figure out how to extrapolate out the lines
    '''
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, 5)

    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    plt.imshow(masked_edges)
    # This time we are defining a four sided polygon to mask

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 60 #minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    plt.imshow(lines_edges)
    return lines_edges



