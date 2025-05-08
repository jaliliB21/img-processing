import cv2
import numpy as np
import math


# 1. Read the image
img = cv2.imread("my_face.jpg")
img = cv2.resize(img, (512, 512))  # Minimize for ease of display


#2. Turn to gray
def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

gray = rgb2gray(img)


# 3. Making a manual Gaussian filter
def gaussian_kernel(size=5, sigma=1.0):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for x in range(size):
        for y in range(size):
            diff = (x - center)**2 + (y - center)**2
            kernel[x, y] = math.exp(-diff / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


# 4. Manual convolution
def apply_filter(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output


# 5. Calculation of the gradient and the angle of the edges
def gradient_magnitude_direction(img):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    Gx = apply_filter(img, Kx)
    Gy = apply_filter(img, Ky)

    mag = np.hypot(Gx, Gy)
    mag = (mag / mag.max()) * 255

    angle = np.arctan2(Gy, Gx)
    angle = np.degrees(angle)
    angle[angle < 0] += 180
    return mag, angle


# 6. Thinning the edges (Non-Max Suppression)
def non_max_suppression(mag, angle):
    Z = np.zeros_like(mag)
    angle = angle % 180
    h, w = mag.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q, r = 255, 255
            angle_ = angle[i, j]

            if (0 <= angle_ < 22.5) or (157.5 <= angle_ <= 180):
                q, r = mag[i, j+1], mag[i, j-1]
            elif 22.5 <= angle_ < 67.5:
                q, r = mag[i+1, j-1], mag[i-1, j+1]
            elif 67.5 <= angle_ < 112.5:
                q, r = mag[i+1, j], mag[i-1, j]
            elif 112.5 <= angle_ < 157.5:
                q, r = mag[i-1, j-1], mag[i+1, j+1]

            if (mag[i, j] >= q) and (mag[i, j] >= r):
                Z[i, j] = mag[i, j]
            else:
                Z[i, j] = 0
    return Z


# --- Execution of steps ---
gaussian = gaussian_kernel(size=5, sigma=1.0)
blurred = apply_filter(gray, gaussian)

mag, angle = gradient_magnitude_direction(blurred)
edges = non_max_suppression(mag, angle).astype(np.uint8)


# Display the results
cv2.imshow("Gray", gray)
cv2.imshow("Blurred", blurred.astype(np.uint8))
cv2.imshow("Edges (Canny-like)", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
