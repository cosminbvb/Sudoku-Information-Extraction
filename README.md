# Sudoku information extraction

### Task 1 - Classic Sudoku

#### 1. Detection

<details>
  
Firstly, the image goes through the following steps:
- Normalization
- Sharpening the image by weigth adding the Median Blurred and Gaussian Blurred versions of the image
- Thresholding
- Eroding (dilates the black borders)
- Canny Edge Detection
  
Then, we plug in the result into the ```cv2.findContours()``` function and determine the contour with maximum area (which represents our sudoku).
  
At this point, we know the 4 corners of the sudoku rectangle. However, the rectangle might be rotated by a small degree to either side. An easy way to crop the sudoku in a straight manner is by calculating the necessary transformation using ```cv2.getPerspectiveTransform()``` and applying it on our image with ```cv2.warpPerspective()```.
  
</details>

#### 2. Cell extraction

<details>
  
Since we know that our final image is of size 500x500 (after resizing, of course),
and our sudoku consists of 9 rows and 9 columns, we can define the inside border lines
which will determine each cell.

Overlaying the lines with the image should look like this:

![](https://github.com/cosminbvb/Sudoku-Information-Extraction/blob/main/documentation/overlay.png)

Then, we iterate through consecutive horizontal and vertical lines to find the bounding points of each cell.
  
</details>



### Task 2 - Jigsaw Sudoku
