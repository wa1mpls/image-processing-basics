import numpy as np
import cv2
from matplotlib import pyplot as plt
import requests
import os
import sys
import io

# Đảm bảo hỗ trợ UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Tải ảnh Lenna nếu chưa tồn tại
url = "http://www.ess.ic.kanagawa-it.ac.jp/std_img/colorimage/Lenna.jpg"
filename = "Lenna.jpg"
if not os.path.exists(filename):
    print(f"Đang tải ảnh {filename}...")
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"File đã được tải thành công: {filename}")
else:
    print(f"Ảnh {filename} đã tồn tại.")

# Hàm đọc ảnh bằng OpenCV
def read_image(file_path):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng grayscale
        return img
    except FileNotFoundError:
        print("Không tìm thấy file ảnh. Hãy kiểm tra lại đường dẫn!")
        exit()

# ========================= Biến đổi màu ========================= #

# Linear Mapping
def linear_mapping(image, a=1.0, b=0):
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            output[i, j] = a * image[i, j] + b
    return np.clip(output, 0, 255).astype(np.uint8)

# Logarithmic Mapping
def log_mapping(image, c=1.0):
    output = c * np.log1p(image.astype(np.float32))
    return np.clip(output, 0, 255).astype(np.uint8)

# Exponential Mapping
def exp_mapping(image, c=1.0):
    output = c * (np.exp(image.astype(np.float32) / 255.0) - 1) * 255
    return np.clip(output, 0, 255).astype(np.uint8)

# Histogram Equalization
def histogram_equalization(image):
    hist = np.zeros(256, dtype=int)
    for pixel in image.ravel():
        hist[pixel] += 1
    cdf = np.cumsum(hist) / np.sum(hist)
    cdf_scaled = (cdf * 255).astype(np.uint8)
    output = cdf_scaled[image]
    return output

# ========================= Biến đổi hình học ========================= #

# Affine Transformation
def affine_transform(image, a, b):
    height, width = image.shape
    output = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            x_new = int(a[0] + a[1]*j + a[2]*i)
            y_new = int(b[0] + b[1]*j + b[2]*i)
            if 0 <= x_new < width and 0 <= y_new < height:
                output[y_new, x_new] = image[i, j]
    return output

# Nearest Neighbor Resize
def nearest_neighbor_resize(image, scale_x, scale_y):
    height, width = image.shape
    new_height = int(height * scale_y)
    new_width = int(width * scale_x)
    output = np.zeros((new_height, new_width), dtype=image.dtype)
    for i in range(new_height):
        for j in range(new_width):
            x = int(j / scale_x)
            y = int(i / scale_y)
            output[i, j] = image[y, x]
    return output
# Linear_interpolation_resize
def linear_interpolation_resize(image, scale_x, scale_y):
    height, width = image.shape
    new_height = int(height * scale_y)
    new_width = int(width * scale_x)
    output = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            # Tọa độ thực trong ảnh gốc
            x = j / scale_x
            y = i / scale_y

            # Các pixel lân cận
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, width - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, height - 1)

            # Hệ số nội suy
            alpha = x - x0
            beta = y - y0

            # Nội suy giá trị pixel
            output[i, j] = (
                (1 - alpha) * (1 - beta) * image[y0, x0] +
                alpha * (1 - beta) * image[y0, x1] +
                (1 - alpha) * beta * image[y1, x0] +
                alpha * beta * image[y1, x1]
            )

    return output
# Bilinear_transform
def bilinear_transform(image, a, b):
    """
    Thực hiện phép biến đổi Bilinear trên ảnh.
    Args:
        image: Ảnh đầu vào (numpy array, grayscale).
        a: Danh sách 4 hệ số [a0, a1, a2, a3] cho công thức x'.
        b: Danh sách 4 hệ số [b0, b1, b2, b3] cho công thức y'.
    Returns:
        Ảnh sau khi biến đổi (numpy array).
    """
    height, width = image.shape
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            # Tọa độ mới sau biến đổi Bilinear
            x_new = a[0] + a[1]*j + a[2]*i + a[3]*j*i
            y_new = b[0] + b[1]*j + b[2]*i + b[3]*j*i

            # Lấy giá trị pixel từ ảnh gốc (Nearest Neighbor)
            x_new_int = int(np.round(x_new))
            y_new_int = int(np.round(y_new))

            # Đảm bảo tọa độ mới nằm trong giới hạn của ảnh
            if 0 <= x_new_int < width and 0 <= y_new_int < height:
                output[i, j] = image[y_new_int, x_new_int]

    return output


# ========================= Làm mịn ảnh (image smoothing) ========================= #

# Averaging Filter
def averaging_filter(image, kernel_size):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            region = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            output[i-pad, j-pad] = np.sum(region) / (kernel_size * kernel_size)
    return output

# Gaussian Filter
def gaussian_kernel(size, sigma):
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def gaussian_filter(image, kernel):
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            region = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            output[i-pad, j-pad] = np.sum(region * kernel)
    return output

# Median Filter
def median_filter(image, kernel_size):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)
    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            region = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            output[i-pad, j-pad] = np.median(region)
    return output

# ========================= Làm mịn ảnh (image blurring) ========================= #

def gaussian_blur(image, kernel):
    """
    Áp dụng Gaussian Blur cho ảnh.
    Args:
        image: Ảnh đầu vào (numpy array, grayscale).
        kernel: Kernel Gaussian (numpy array).
    Returns:
        Ảnh đã làm mờ (numpy array).
    """
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            region = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            output[i-pad, j-pad] = np.sum(region * kernel)

    return output


# ========================= Menu Chọn ========================= #
def safe_input(prompt):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return sys.stdin.readline().strip()

def menu():
    print("\nChọn chức năng:")
    print("1. Biến đổi màu")
    print("2. Biến đổi hình học")
    print("3. Làm mịn ảnh")
    print("4. Thoát")
    return int(input("Nhập lựa chọn của bạn: "))

def color_transform_menu(image):
    print("\nChọn loại biến đổi màu:")
    print("1. Linear Mapping")
    print("2. Logarithmic Mapping")
    print("3. Exponential Mapping")
    print("4. Histogram Equalization")
    choice = int(input("Nhập lựa chọn: "))
    if choice == 1:
        a = float(input("Nhập hệ số a: "))
        b = int(input("Nhập hệ số b: "))
        result = linear_mapping(image, a, b)
    elif choice == 2:
        c = float(input("Nhập hệ số c: "))
        result = log_mapping(image, c)
    elif choice == 3:
        c = float(input("Nhập hệ số c: "))
        result = exp_mapping(image, c)
    elif choice == 4:
        result = histogram_equalization(image)
    else:
        print("Lựa chọn không hợp lệ.")
        return
    plt.imshow(result, cmap='gray')
    plt.title("Kết quả")
    plt.show()

def geometric_transform_menu(image):
    print("\nChọn loại biến đổi hình học:")
    print("1. Affine Transformation")
    print("2. Resize (Nearest Neighbor)")
    print("3. Resize (Linear Interpolation)")
    print("4. Bilinear Transform")  # Tùy chọn mới
    choice = int(input("Nhập lựa chọn: "))
    if choice == 1:
        a = list(map(float, input("Nhập tham số a (3 số): ").split()))
        b = list(map(float, input("Nhập tham số b (3 số): ").split()))
        result = affine_transform(image, a, b)
    elif choice == 2:
        scale_x = float(input("Nhập hệ số scale_x: "))
        scale_y = float(input("Nhập hệ số scale_y: "))
        result = nearest_neighbor_resize(image, scale_x, scale_y)
    elif choice == 3:
        scale_x = float(input("Nhập hệ số scale_x: "))
        scale_y = float(input("Nhập hệ số scale_y: "))
        result = linear_interpolation_resize(image, scale_x, scale_y)
    elif choice == 4:  # Xử lý Bilinear Transform
        a = list(map(float, input("Nhập tham số a (4 số): ").split()))
        b = list(map(float, input("Nhập tham số b (4 số): ").split()))
        result = bilinear_transform(image, a, b)
    else:
        print("Lựa chọn không hợp lệ.")
        return
    plt.imshow(result, cmap='gray')
    plt.title("Kết quả")
    plt.show()



def smoothing_menu(image):
    print("\nChọn loại làm mịn ảnh:")
    print("1. Averaging Filter")
    print("2. Gaussian Filter (Custom Kernel)")
    print("3. Median Filter")
    print("4. Gaussian Blur (Predefined Kernel)")  # Thêm tùy chọn Gaussian Blur
    choice = int(input("Nhập lựa chọn: "))
    if choice == 1:
        kernel_size = int(input("Nhập kích thước kernel: "))
        result = averaging_filter(image, kernel_size)
    elif choice == 2:
        kernel_size = int(input("Nhập kích thước kernel: "))
        sigma = float(input("Nhập sigma: "))
        kernel = gaussian_kernel(kernel_size, sigma)
        result = gaussian_filter(image, kernel)
    elif choice == 3:
        kernel_size = int(input("Nhập kích thước kernel: "))
        result = median_filter(image, kernel_size)
    elif choice == 4:  # Xử lý Gaussian Blur
        kernel_size = int(input("Nhập kích thước kernel (lẻ, ví dụ 3, 5, 7): "))
        sigma = float(input("Nhập sigma: "))
        kernel = gaussian_kernel(kernel_size, sigma)
        result = gaussian_blur(image, kernel)
    else:
        print("Lựa chọn không hợp lệ.")
        return
    plt.imshow(result, cmap='gray')
    plt.title("Kết quả")
    plt.show()

# Chạy chương trình chính
def main():
    image_path = "Lenna.jpg"
    image = read_image(image_path)
    while True:
        choice = menu()
        if choice == 1:
            color_transform_menu(image)
        elif choice == 2:
            geometric_transform_menu(image)
        elif choice == 3:
            smoothing_menu(image)
        elif choice == 4:
            print("Thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()
