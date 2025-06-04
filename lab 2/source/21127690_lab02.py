import cv2
import numpy as np
import time
import requests
import os
import matplotlib.pyplot as plt

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

def read_image(image_path='Lenna.jpg'):
    """ Đọc ảnh và chuyển thành ảnh xám """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# ================================
# 1. Gradient Operators
# ================================


def sobel_operator(image):
    # 1. Làm mịn ảnh bằng Gaussian để giảm nhiễu
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 2. Kernel cho Sobel (trục x và trục y)
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])

    # 3. Tính Gx và Gy
    Gx = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_x)
    Gy = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_y)

    # 4. Tính độ lớn gradient
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # 5. Chuẩn hóa gradient về khoảng [0, 255] để hiển thị dưới dạng edges
    edges = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return edges


def prewitt_operator(image):
    """ Tự code Prewitt Operator """
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)
    edges = np.sqrt(edges_x ** 2 + edges_y ** 2)
    return edges


def robert_operator(image):
    """ Tự code Robert Operator """
    h, w = image.shape
    edges = np.zeros_like(image)
    for y in range(h - 1):
        for x in range(w - 1):
            gx = int(image[y, x]) - int(image[y + 1, x + 1])
            gy = int(image[y + 1, x]) - int(image[y, x + 1])
            edges[y, x] = np.sqrt(gx ** 2 + gy ** 2)
    return edges


def frei_chen_operator(image):
    """ Tự code Frei-Chen Operator """
    sqrt2 = np.sqrt(2)
    kernel_x = np.array([[-1, 0, 1], [-sqrt2, 0, sqrt2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -sqrt2, -1], [0, 0, 0], [1, sqrt2, 1]])
    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)
    edges = np.sqrt(edges_x ** 2 + edges_y ** 2)
    return edges


# ================================
# 2. Laplace Operators
# ================================

def laplace_operator(image):
    """ Tự code Laplace Operator """
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    edges = cv2.filter2D(image, -1, kernel)
    return edges


# ================================
# 3. Laplace of Gaussian (LoG)
# ================================

def laplace_of_gaussian(image):
    """ Tự code Laplace of Gaussian (LoG) """
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
    edges = laplace_operator(blurred)
    return edges


# ================================
# 4. Canny (7 bước đầy đủ)
# ================================

def gaussian_blur(image, kernel_size=9, sigma=1.4):
    """ Tự code Gaussian Blur """
    k = kernel_size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    padded_image = np.pad(image, ((k, k), (k, k)), mode='constant', constant_values=0)
    h, w = image.shape
    output = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            region = padded_image[y:y+kernel_size, x:x+kernel_size]
            output[y, x] = np.sum(region * kernel)
    return output


def sobel_operator_forcanny(image):
    """ Tự code Sobel Operator """
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], 
                         [0, 0, 0], 
                         [1, 2, 1]])
    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)
    edges = np.sqrt(edges_x ** 2 + edges_y ** 2)
    return edges, edges_x, edges_y


def non_maximum_suppression(magnitude, angle):
    """ Làm mảnh biên """
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float32)
    angle = angle * 180.0 / np.pi
    angle[angle < 0] += 180

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            q, r = 255, 255
            if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] <= 180):
                q = magnitude[y, x + 1]
                r = magnitude[y, x - 1]
            elif (22.5 <= angle[y, x] < 67.5):
                q = magnitude[y + 1, x - 1]
                r = magnitude[y - 1, x + 1]
            elif (67.5 <= angle[y, x] < 112.5):
                q = magnitude[y + 1, x]
                r = magnitude[y - 1, x]
            elif (112.5 <= angle[y, x] < 157.5):
                q = magnitude[y - 1, x - 1]
                r = magnitude[y + 1, x + 1]
            
            if (magnitude[y, x] >= q) and (magnitude[y, x] >= r):
                nms[y, x] = magnitude[y, x]
            else:
                nms[y, x] = 0

    return nms


def double_threshold(nms, low_threshold, high_threshold):
    """ Ngưỡng hóa biên kép """
    strong = 255
    weak = 50
    strong_edges = (nms >= high_threshold)
    weak_edges = ((nms <= high_threshold) & (nms >= low_threshold))
    result = np.zeros_like(nms, dtype=np.uint8)
    result[strong_edges] = strong
    result[weak_edges] = weak
    return result

def select_thresholds(image):
    """ Chọn ngưỡng thích hợp cho thuật toán Canny """
    # Vẽ histogram của ảnh
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.title("Histogram of Image")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    # Hướng dẫn người dùng chọn ngưỡng thủ công
    print("Xem histogram và chọn ngưỡng thấp và ngưỡng cao.")
    low_threshold = int(input("Nhập ngưỡng thấp (low_threshold): "))
    high_threshold = int(input("Nhập ngưỡng cao (high_threshold): "))
    
    return low_threshold, high_threshold


def edge_tracking_by_hysteresis(result):
    """ Theo dõi biên bằng ngưỡng hóa """
    h, w = result.shape
    strong = 255
    weak = 50

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if result[y, x] == weak:
                if (strong in result[y-1:y+2, x-1:x+2]):
                    result[y, x] = strong
                else:
                    result[y, x] = 0
    return result


def feature_synthesis(edges_list):
    """ Tổng hợp các thông tin từ nhiều tỷ lệ """
    return np.max(np.array(edges_list), axis=0)


def canny_custom(image, sigma_values=[1.0, 1.4, 2.0, 2.4, 3.0]):
    """ Cài đặt đầy đủ 7 bước của Canny với ngưỡng chỉ chọn một lần """
    edges_list = []
    
    # Bước 1: Chọn ngưỡng chỉ 1 lần (sử dụng sigma đầu tiên)
    sigma = sigma_values[0]
    print(f"Chọn ngưỡng bằng sigma = {sigma}")
    
    # Giảm nhiễu và tính biên bước đầu
    blurred = gaussian_blur(image, kernel_size=9, sigma=sigma)
    magnitude, gx, gy = sobel_operator_forcanny(blurred)
    angle = np.arctan2(gy, gx)
    nms = non_maximum_suppression(magnitude, angle)
    
    # Kiểm tra xem NMS đã hoạt động chưa
    if np.max(nms) == 0:
        print("Cảnh báo: Không có biên phát hiện sau Non-Maximum Suppression. Kiểm tra lại ảnh đầu vào hoặc tham số.")
    
    # Chọn ngưỡng chỉ 1 lần
    low_threshold, high_threshold = select_thresholds(nms)
    print(f"Ngưỡng đã chọn: low_threshold = {low_threshold}, high_threshold = {high_threshold}")
    
    # Áp dụng quy trình Canny với tất cả các sigma
    for sigma in sigma_values:
        print(f"Xử lý với sigma = {sigma}")
        
        # Bước 1: Giảm nhiễu bằng Gaussian Blur
        blurred = gaussian_blur(image, kernel_size=5, sigma=sigma)
        
        # Bước 2: Tính gradient (Sobel)
        magnitude, gx, gy = sobel_operator_forcanny(blurred)
        angle = np.arctan2(gy, gx)
        
        # Bước 3: Làm mảnh biên (Non-Maximum Suppression)
        nms = non_maximum_suppression(magnitude, angle)

        if np.max(nms) == 0:
            print("Cảnh báo: Không có biên phát hiện sau Non-Maximum Suppression. Kiểm tra lại ảnh đầu vào hoặc tham số.")
        
        # Bước 4: Ngưỡng hóa biên kép (sử dụng ngưỡng đã chọn từ đầu)
        result = double_threshold(nms, low_threshold, high_threshold)
        
        # Bước 5: Theo dõi biên (Edge Tracking by Hysteresis)
        result = edge_tracking_by_hysteresis(result)
        
        edges_list.append(result)
    
    # Bước 6 & 7: Lặp lại và tổng hợp thông tin từ nhiều tỷ lệ
    combined_edges = feature_synthesis(edges_list)
    
    # Kiểm tra lại ảnh kết quả tổng hợp
    if np.max(combined_edges) == 0:
        print("Cảnh báo: Không có biên trong kết quả cuối cùng. Kiểm tra tham số ngưỡng.")
    
    return combined_edges




def measure_time(function, image):
    start_time = time.time()
    result = function(image)
    end_time = time.time()
    return result, end_time - start_time


# ================================
# Menu và Xử lý lựa chọn
# ================================

def menu():
    print("\nChọn loại toán tử phát hiện biên:")
    print("1. Gradient Operator")
    print("2. Laplace")
    print("3. Laplace of Gaussian (LoG)")
    print("4. Canny")
    print("5. Thoát")
    return int(input("Nhập lựa chọn của bạn (1-5): "))


def gradient_menu():
    print("\nChọn toán tử Gradient:")
    print("1. Sobel")
    print("2. Prewitt")
    print("3. Robert")
    print("4. Frei-Chen")
    return int(input("Nhập lựa chọn của bạn (1-4): "))

def main():
    image = cv2.imread('Lenna.jpg', cv2.IMREAD_GRAYSCALE)
    while True:
        choice = menu()
        if choice == 5:
            break

        if choice == 1:  # Gradient Operator
            operator_choice = gradient_menu()
            operators = ['sobel', 'prewitt', 'robert', 'frei_chen']
            
            if 1 <= operator_choice <= 4:  # Ensure valid input for operator
                operator_name = operators[operator_choice - 1]
                
                #if method_choice == 1:  # Tự code
                if operator_name == 'sobel':
                        edges = sobel_operator(image)
                elif operator_name == 'prewitt':
                        edges = prewitt_operator(image)
                elif operator_name == 'robert':
                        edges = robert_operator(image)
                elif operator_name == 'frei_chen':
                        edges = frei_chen_operator(image)
                #elif method_choice == 2:  # Dùng OpenCV
                if operator_name == 'sobel':
                        edges_cv = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
                elif operator_name == 'prewitt':
                        edges = prewitt_operator(image)  # OpenCV doesn't have Prewitt
                elif operator_name == 'robert':
                        edges = robert_operator(image)  # OpenCV doesn't have Robert
                elif operator_name == 'frei_chen':
                        edges = frei_chen_operator(image)  # OpenCV doesn't have Frei-Chen

                # Hiển thị kết quả
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 3, 1)
                plt.imshow(image, cmap='gray')
                plt.title('Ảnh gốc')

                plt.subplot(1, 3, 2)
                plt.imshow(edges, cmap='gray')
                plt.title(f"{operator_name} (Tự code)")

                if operator_name == 'sobel':
                    plt.subplot(1, 3, 3)
                    plt.imshow(edges_cv, cmap='gray')
                    plt.title('Sobel (OpenCV)')
            
                plt.show()

        elif choice == 2:  # Laplace
            edges = laplace_operator(image)
            edges_cv = cv2.Laplacian(image, cv2.CV_64F)
            
            # Hiển thị kết quả
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Ảnh gốc')

            plt.subplot(1, 3, 2)
            plt.imshow(edges, cmap='gray')
            plt.title('Laplace (Tự code)')

            plt.subplot(1, 3, 3)
            plt.imshow(edges_cv, cmap='gray')
            plt.title('Laplace (OpenCV)')
                    
            plt.show()

        elif choice == 3:  # Laplace of Gaussian (LoG)
            edges = laplace_of_gaussian(image)
            # Áp dụng Gaussian Blur
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
            # Áp dụng Laplacian
            edges_cv = cv2.Laplacian(blurred_image, cv2.CV_64F)
            edges_cv = np.uint8(np.absolute(edges_cv))

            # Hiển thị kết quả
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Ảnh gốc')

            plt.subplot(1, 3, 2)
            plt.imshow(edges, cmap='gray')
            plt.title('Laplace of Gaussian (Tự code)')

            plt.subplot(1, 3, 3)
            plt.imshow(edges_cv, cmap='gray')
            plt.title('Laplace of Gaussian (OpenCV)')
                    
            plt.show()

        elif choice == 4:  # Canny
            edges = canny_custom(image)
            # Áp dụng Canny Edge Detection
            edges_cv = cv2.Canny(image, 100, 200)  # Tham số có thể điều chỉnh (minVal, maxVal)
            # Hiển thị kết quả
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Ảnh gốc')

            plt.subplot(1, 3, 2)
            plt.imshow(edges, cmap='gray')
            plt.title('Canny (Tự code)')

            plt.subplot(1, 3, 3)
            plt.imshow(edges_cv, cmap='gray')
            plt.title('Canny (OpenCV)')
        
            plt.show()

# Run the program
main()
