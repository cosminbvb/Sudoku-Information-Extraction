# Sudoku information extraction

Given an image containing a sudoku, extract all the relevant information required to solve it.
This repository consists of two tasks: extracting the information in **both classic and jigsaw sudoku**

## Task 1 - Classic Sudoku
* [0. How to run?](#run)
* [1. Detection](#detection)
* [2. Cell extraction](#cell_extraction)
* [3. Cell content classification](#cell_content_classification)

## Task 2 - Jigsaw Sudoku
* [0. How to run?](#run)
* [1. Detection](#detection)
* [2. Region extraction](#region_extraction)
* [3. Cell extraction](#cell_extraction)
* [4. Cell content classification](#cell_content_classification)

## Common functions
The following functions are used in both tasks and in order to avoid duplicate code they were moved into ```helpers/utils.py```.
* show_image(title, image)
* normalize(image)
* preprocess_image(image)
* detect_sudoku(image)
* extract_cells(image)

<a name="run"/>

### How to run?

Call the function ``` get_results(input_dir, output_dir, number_of_samples) ``` where:

- input_dir refers to the directory in which your original sudoku images are located.
- output_dir refers to the directory in which you want your results
- number_of_samples on which you want to run inference

Example:
```
input_dir = 'datasets/antrenare/clasic'
output_dir = 'results'
number_of_samples = 20  # number of input images
get_results(input_dir, output_dir, number_of_samples)
```
*Make sure you are using the same image naming convension (0{x}.jpg if x < 10 otherwise {x}.jpg) or change the names in ``` get_results()``` .


<a name="detection"/>

### Detection (extracting the sudoku square)

Firstly, the image goes through the following steps:
- Normalization
- Sharpening the image by weigth adding the Median Blurred and Gaussian Blurred versions of the image
- Thresholding
- Eroding (dilates the black borders)
- Canny Edge Detection
  
Then, we plug in the result into the ```cv2.findContours()``` function and determine the contour with maximum area (which represents our sudoku).
  
At this point, we know the 4 corners of the sudoku rectangle. However, the rectangle might be rotated by a small degree to either side. An easy way to crop the sudoku in a straight manner is by calculating the necessary transformation using ```cv2.getPerspectiveTransform()``` and applying it on our image with ```cv2.warpPerspective()```.


<a name="cell_extraction"/>

### Cell extraction

Since we know that our final image is of size 500x500 (after resizing, of course),
and our sudoku consists of 9 rows and 9 columns, we can define the inside border lines
which will determine each cell.

Overlaying the lines with the image should look like this:

![](https://github.com/cosminbvb/Sudoku-Information-Extraction/blob/main/screenshots/overlay.png)

Then, we iterate through consecutive horizontal and vertical lines to find the bounding points of each cell.

<a name="region_extraction"/>

### Region extraction

This sub-task consists of finding the bold lines which define the regions. Due to the sample variation, I found it quite ineffective to apply filters and morphological operators that perfectly eliminate the thin lines while also keeping all the bold ones for each sample, even after normalization, thresholding, blurring, etc. Therefore, I chose to
classify a line as either thin or bold by doing the following:
* converting the image to grayscale and normalizing
* computing the pixel sum of all the lines
* storing a minimum and a maximum sum (meaning the boldest line and the thinnest line)
* if the pixel sum of our current line is closer to the sum of the boldest line, then it's bold, otherwise it's thin

Note: This approach scored 40 / 40.

Note: This approach uses a sliding-window approach with an error (padding) of 5px.

After determining the bold lines, I stored them into two matrices:
```vertical_lines``` and ```horizontal_lines```.

Finally, to determine the regions, I started a BFS from each yet unvisited cell and marked all the newly visited cells with a region number (by calling ```fill_region(start_i, start_j, region_number...)```). Before adding a neighbour in the queue, the function checks ```vertical_lines``` and ```horizontal_lines``` for illegal moves.

TODO: 1.jpg -> region matrix

<a name="cell_content_classification"/>

### Cell content classification

This sub-task consists of classifying the content of a cell, where classes range from 0 (empty cell) to 9.

My approach was building and training a Convolutional Neural Network (the code can be found [here](https://github.com/cosminbvb/Sudoku-Information-Extraction/blob/main/cell_classification.ipynb)).
 
The dataset I used was made for this exact task and contains around 3k samples. 
You can find it on Kaggle: https://www.kaggle.com/kshitijdhama/printed-digits-dataset
