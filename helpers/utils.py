import cv2 as cv
import numpy as np

def show_image(title,image):
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def normalize(img):
    noise = cv.dilate(img, np.ones((7,7),np.uint8))
    blur = cv.medianBlur(noise, 21)
    res = 255 - cv.absdiff(img, blur)
    no_shdw = cv.normalize(res,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    return no_shdw 

def preprocess_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    normalized = normalize(image)

    image_m_blur = cv.medianBlur(normalized, 3)
    # kernel_size = 3

    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 7)
    # kernel_size = (0, 0) (if it s 0, it s computed from sigma)
    # std_deviation = 5

    image_sharpened = cv.addWeighted(image_m_blur, 1.9, image_g_blur, -0.9, 0)
    # last arg = scalar added to each sum

    _, thresh = cv.threshold(image_sharpened, 180, 255, cv.THRESH_BINARY)
    # threshold = 180
    # maxValue = 255 (every pixel > threshold becomes maxValue and the rest become 0)

    # Adaptive Gaussian Thresholding is worth trying (ofc, not exactly in this context) but made it work without it
    # thresh = cv.adaptiveThreshold(image_sharpened, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv.THRESH_BINARY, 11, 2)
    # maxValue = 255, blockSize = 11 (size of a pixel neighborhood),
    # C = 2 (constant substracted from the mean or weighted mean)

    kernel = np.ones((4, 4), np.uint8)

    eroded = cv.erode(thresh, kernel)

    # I also applied the Sobel filter to get a look
    sobelX = cv.Sobel(eroded, ddepth=cv.CV_32F, dx=1, dy=0)
    sobelY = cv.Sobel(eroded, ddepth=cv.CV_32F, dx=0, dy=1)
    sobelX = cv.convertScaleAbs(sobelX)
    sobelY = cv.convertScaleAbs(sobelY)
    # combine the gradient representations into a single image
    sobel = cv.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    # just to take a look at the gradients
    # print(labels)

    edges = cv.Canny(eroded, 20, 150)
    # threshold1 = 20 and threshold2 = 150 are used for the hysteresis procedure

    # # Displaying each process result:
    # show_image("original", image)
    # show_image("normalized", normalized)
    # show_image("median blurred", image_m_blur)
    # show_image("gaussian blurred", image_g_blur)
    # show_image("sharpened", image_sharpened)
    # show_image("threshold of blur", thresh)
    # show_image("eroded", eroded)
    # show_image("sobel", sobel)
    # show_image("canny", edges)

    contours, _ = cv.findContours(
        edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    max_area = 0

    for i in range(len(contours)):
        if(len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left], [possible_top_right], [
                                          possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    # # Displaying the 4 corners:
    # image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    # cv.circle(image_copy, tuple(top_left), 4, (0, 0, 255), -1)
    # cv.circle(image_copy, tuple(top_right), 4, (0, 0, 255), -1)
    # cv.circle(image_copy, tuple(bottom_left), 4, (0, 0, 255), -1)
    # cv.circle(image_copy, tuple(bottom_right), 4, (0, 0, 255), -1)
    # show_image("detected corners", image_copy)

    corners = np.asarray([top_left, top_right, bottom_left, bottom_right], dtype="float32")
    return corners

def detect_sudoku(img):
    corners = preprocess_image(img.copy())
    top_left, top_right, bottom_left, bottom_right = corners

    # calculating the length of each edge and picking the maximum
    # width and height
    width_bottom = np.sqrt(((bottom_right[0] - bottom_left[0])
                            ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_top = np.sqrt(((top_right[0] - top_left[0]) ** 2) +
                        ((top_right[1] - top_left[1]) ** 2))
    width = max(int(width_top), int(width_bottom))

    height_right = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) +
                           ((top_right[1] - bottom_right[1]) ** 2))
    height_left = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) +
                          ((top_left[1] - bottom_left[1]) ** 2))
    height = max(int(height_left), int(height_right))

    dimensions = np.array([[0, 0], [width - 1, 0], [0, height - 1],
                          [width - 1, height - 1]], dtype="float32")

    transform = cv.getPerspectiveTransform(corners, dimensions)
    return cv.warpPerspective(img, transform, (width, height))

def extract_cells(img):
    # Since we know that our final image is of size 500x500 (after resizing, of course),
    # and our sudoku consists of 9 rows and 9 columns, we can define the inside border lines
    # which will determine each cell
    lines_vertical = []
    lines_horizontal = []
    for i in range(0, 500, 55):
        lines_vertical.append([(i, 0), (i, 499)])
        lines_horizontal.append([(0, i), (499, i)])

    cells = []
    padding = 10 # padding each cell with 5px to make sure we don't crop out the lines too
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + padding
            y_max = lines_vertical[j + 1][1][0] - padding
            x_min = lines_horizontal[i][0][1] + padding
            x_max = lines_horizontal[i + 1][1][1] - padding
            cell = img[x_min:x_max, y_min:y_max].copy()
            cells.append(cv.resize(cell, (28, 28)))  # resizing to 28x28 to match the cnn input size
    return np.array(cells)