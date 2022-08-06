import cv2
import glob

imdir = 'D:/01test/Dataset/Test/Pa Tangke Lumu/'
ext = ['png', 'jpg', 'gif']    # Add image formats here

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

images = [cv2.imread(file) for file in files]
    
# ddept=cv2.CV_16S
# scale=1
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ref, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     x = cv2.Sobel(gray, ddept, 1,0, ksize=3, scale=1)
#     y = cv2.Sobel(gray, ddept, 0,1, ksize=3, scale=1)
#     absx= cv2.convertScaleAbs(x)
#     absy = cv2.convertScaleAbs(y)
#     grad = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
#     cv2.imshow('edge', grad)
#     cv2.imshow('main', frame)
#     if cv2.waitKey(1)==27:
#         break
# cv2.destroyAllWindows()
# cap.release()

# ddept=cv2.CV_16S  
# # img2 = cv2.imread('D:/01test/flask app/assets/images/upload/upload.jpg')
# gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
# x = cv2.Sobel(gray, ddept, 1,0, ksize=3, scale=1)
# y = cv2.Sobel(gray, ddept, 0,1, ksize=3, scale=1)
# absx= cv2.convertScaleAbs(x)
# absy = cv2.convertScaleAbs(y)
# edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
# # cv2.imwrite('D:/01test/flask app/assets/images/upload/canny.jpg', edge)

    #   def sHalf(T, sigma):
    #      temp = -np.log(T) * 2 * (sigma ** 2)
    #      return np.round(np.sqrt(temp))

    #   def calculate_filter_size(T, sigma):
    #      return 2*sHalf(T, sigma) + 1
         
    #   def MaskGeneration(T, sigma):
    #      N = calculate_filter_size(T, sigma)
    #      shalf = sHalf(T, sigma)
    #      y, x = np.meshgrid(range(-int(shalf), int(shalf) + 1), range(-int(shalf), int(shalf) + 1))
    #      return x, y

    #   def Gaussian(x,y, sigma):
    #      temp = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
    #      return (np.exp(-temp))

    #   def calculate_gradient_X(x,y, sigma):
    #      temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    #      return -((x * np.exp(-temp)) / sigma ** 2)

    #   def calculate_gradient_Y(x,y, sigma):
    #      temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    #      return -((y * np.exp(-temp)) / sigma ** 2)

    #   def pad(img, kernel):
    #      r, c = img.shape
    #      kr, kc = kernel.shape
    #      padded = np.zeros((r + kr,c + kc), dtype=img.dtype)
    #      insert = np.uint((kr)/2)
    #      padded[insert: insert + r, insert: insert + c] = img
    #      return padded
                  
    #   def smooth(img, kernel=None):
    #      if kernel is None:
    #         mask = np.array([[1,1,1],[1,1,1],[1,1,1]])
    #      else:
    #         mask = kernel
    #      i, j = mask.shape
    #      output = np.zeros((img.shape[0], img.shape[1]))           
    #      image_padded = pad(img, mask)
    #      for x in range(img.shape[0]):    
    #         for y in range(img.shape[1]):
    #               output[x, y] = (mask * image_padded[x:x+i, y:y+j]).sum() / mask.sum()  
    #      return output

    #   def Create_Gx(fx, fy):
    #      gx = calculate_gradient_X(fx, fy, sigma)
    #      gx = (gx * 255)
    #      return np.around(gx)

    #   def Create_Gy(fx, fy):    
    #      gy = calculate_gradient_Y(fx, fy, sigma)
    #      gy = (gy * 255)
    #      return np.around(gy)

    #   def ApplyMask(image, kernel):
    #      i, j = kernel.shape
    #      kernel = np.flipud(np.fliplr(kernel))    
    #      output = np.zeros_like(image)           
    #      image_padded = pad(image, kernel)
    #      for x in range(image.shape[0]):    
    #         for y in range(image.shape[1]):
    #               output[x, y] = (kernel * image_padded[x:x+i, y:y+j]).sum()        
    #      return output

    #   def Gradient_Magnitude(fx, fy):
    #      mag = np.zeros((fx.shape[0], fx.shape[1]))
    #      mag = np.sqrt((fx ** 2) + (fy ** 2))
    #      mag = mag * 100 / mag.max()
    #      return np.around(mag)

    #   def Gradient_Direction(fx, fy):
    #      g_dir = np.zeros((fx.shape[0], fx.shape[1]))
    #      g_dir = np.rad2deg(np.arctan2(fy, fx)) + 180
    #      return g_dir

    #   sigma = 0.5
    #   T = 0.3
    #   x, y = MaskGeneration(T, sigma)
    #   gauss = Gaussian(x, y, sigma)

    #   gx = -Create_Gx(x, y)
    #   gy = -Create_Gy(x, y)

    #   image = cv2.imread('D:/01test/flask app/assets/images/upload/upload.jpg')
    #   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #   # plt.figure(figsize = (8,8))
    #   # plt.imshow(gray, cmap='gray')
    #   # plt.show()

    #   smooth_img = smooth(gray, gauss)
    #   # plt.figure(figsize = (8,8))
    #   # plt.imshow(smooth_img, cmap='gray')

    #   fx = ApplyMask(smooth_img, gx)
    #   # plt.figure(figsize = (8,8))
    #   # plt.imshow(fx, cmap='gray')
      
    #   fy = ApplyMask(smooth_img, gy)
    #   # plt.figure(figsize = (8,8))
    #   # plt.imshow(fy, cmap='gray')

    #   mag = Gradient_Magnitude(fx, fy)
    #   mag = mag.astype(int)
    #   cv2.imwrite('D:/01test/flask app/assets/images/upload/canny.jpg', mag)
    #   filename2 = "canny.jpg"  
    #   file_path2 = os.path.join('D:/01test/flask app/assets/images/upload/', filename2)
    #   # plt.figure(figsize = (18,18))
    #   # plt.imshow(mag, cmap='gray')

    #   # print('max', mag.max())
    #   # print('min', mag.min())