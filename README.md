# Sudoku information extraction

Given an image containing a sudoku, extract all the relevant information required to solve it.
This repository consists of two tasks: extracting the information in **both classic and jigsaw sudoku**.

Note: Each task also required two types of cell content classification: binary (empty / non-empty) and digit (0-9 where 0 represents an empty cell).

## Task 1 - Classic Sudoku
* [0. How to run?](#run)
* [1. Detection](#detection)
* [2. Cell extraction](#cell_extraction)
* [3. Cell content binary classification](#cell_content_binary_classification)
* [4. Cell content digit classification](#cell_content_digit_classification)
## Task 2 - Jigsaw Sudoku
* [0. How to run?](#run)
* [1. Detection](#detection)
* [2. Region extraction](#region_extraction)
* [3. Cell extraction](#cell_extraction)
* [4. Cell content binary classification](#cell_content_binary_classification)
* [5. Cell content digit classification](#cell_content_digit_classification)

## Common functions
The following functions are used in both tasks and in order to avoid duplicate code they were moved into ```helpers/utils.py```.
* ```show_image(title, image)```
* ```normalize(image)```
* ```preprocess_image(image)``` - returns the coordinates of the sudoku rectangle
* ```detect_sudoku(image)``` - returns the cropped and straightened sudoku rectangle
* ```extract_cells(image)``` - returns an array of 81 28x28 images (cropped cells)
* ```get_binary_labels(image)``` - returns 81 binary labels
* ```get_digit_labels(image, model)``` - returns 81 0-9 labels
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

This sub-task consists of finding the bold lines which define the regions. Due to the sample variation, I found it quite ineffective to apply filters and morphological operators that perfectly eliminate the thin lines while also keeping all the bold ones for each sample, even after normalization, thresholding, blurring, etc. Therefore, I chose to find the threshold separating the thin lines by the bold lines by doing the following:
* converting the image to grayscale and normalizing
* computing the mean value of all the lines and storing them in an array
* sorting the array and finding the biggest consecutive difference
* set the threshold as thea mean of the two elements that determined the biggest difference

Note: This approach scored 40 / 40.

Note: This approach uses a sliding-window approach with an error (padding) of 5px.

After determining the bold lines, I stored them into two matrices:
```vertical_lines``` and ```horizontal_lines```.

Finally, to determine the regions, I started a BFS from each yet unvisited cell and marked all the newly visited cells with a region number (by calling ```fill_region(start_i, start_j, region_number...)```). Before adding a neighbour in the queue, the function checks ```vertical_lines``` and ```horizontal_lines``` for illegal moves.

TODO: 1.jpg -> region matrix


<a name="cell_content_binary_classification"/>

### Cell content binary classification

Firstly, I calculated the gradients of each cell with the Sobel filter. Then, I calculated the mean of each cell gradients. Idealy, an empty cell should have a mean close to 0. In order to find the threshold between the empty and non-empty cells I sorted the array with means and calculated the biggest consecutive difference. Once I found it, I set the threshold to to the mean of the two elements that determined the biggest difference.


<a name="cell_content_digit_classification"/>

### Cell content digit classification

This sub-task consists of classifying the content of a cell, where classes range from 0 (empty cell) to 9.

The trained model is saved [here](https://github.com/cosminbvb/Sudoku-Information-Extraction/blob/main/saved_model/model.h5).

My approach was building and training a Convolutional Neural Network (the code can be found [here](https://github.com/cosminbvb/Sudoku-Information-Extraction/blob/main/cell_digit_classification.ipynb)).
 
The dataset I used was made for this exact task and contains around 3k samples. 
You can find it on Kaggle: https://www.kaggle.com/kshitijdhama/printed-digits-dataset
