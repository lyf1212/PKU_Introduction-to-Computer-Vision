import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type=="zeroPadding":
        h, w = img.shape[0], img.shape[1]
        h_ = h + 2 * padding_size 
        w_ = w + 2 * padding_size
        padding_img = np.zeros((h_, w_), dtype=float)
        padding_img[padding_size : padding_size + h, padding_size : padding_size + w] = img
        
        return padding_img
        
    elif type=="replicatePadding":
        h, w = img.shape[0], img.shape[1]
        h_ = h + 2 * padding_size 
        w_ = w + 2 * padding_size
        padding_img = np.zeros((h_, w_), dtype=float)
        padding_img[:padding_size, :padding_size] = img[0][0]
        padding_img[h + padding_size: h_, w + padding_size: w_] = img[h-1][w-1]
        padding_img[:padding_size, w + padding_size: w_] = img[0][w-1]
        padding_img[h + padding_size: h_, :padding_size] = img[h-1][0]

        return padding_img
    
    elif type=="reflectionPadding":
        h, w = img.shape[0], img.shape[1]
        h_ = h + 2 * padding_size 
        w_ = w + 2 * padding_size
        
        # 填充四个角部：对二维矩阵每个轴都做一次反向，使用[::-1]方法.
        padding_img[:padding_size, :padding_size] = img[1 : padding_size + 1, 1 : padding_size + 1][::-1, ::-1]
        padding_img[h + padding_size : h_, :padding_size] = img[h - padding_size - 1: h - 1, 1 : padding_size + 1][::-1, ::-1]
        padding_img[:padding_size, w + padding_size : w_] = img[1 : padding_size + 1, w - padding_size - 1: w - 1][::-1, ::-1]
        padding_img[h + padding_size :h_, w + padding_size : w_] = img[h - padding_size - 1:h - 1, w - padding_size - 1:w - 1][::-1, ::-1]
        
        # 填充中间的原始数据.
        padding_img[padding_size : h + padding_size, padding_size : w + padding_size] = img
        
        # 填充剩下的四个边.
        padding_img[:padding_size, padding_size : w + padding_size] = img[1 : padding_size + 1, :][::-1, :]
        padding_img[padding_size : padding_size + h, :padding_size] = img[:, 1 : padding_size + 1][:, ::-1]
        padding_img[h + padding_size:, padding_size : w + padding_size] = img[h - padding_size - 1 : h - 1, :][::-1, :]
        padding_img[padding_size : h + padding_size, w + padding_size:] = img[:, w - padding_size - 1 : w - 1][:, ::-1]
        
        return padding_img



def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img = padding(img, 1, 'zeroPadding')
    padding_img = padding_img.reshape(64, 1)
    
    #build the Toeplitz matrix and compute convolution
    Toeplitz = np.zeros(64).reshape(1, 64)
    
    # It's really important to notice this indexing method.
    Toeplitz[np.array([0,1,2,8,9,10,16,17,18])] = kernel.reshape(-1)[:]
    Toeplitz = np.repeat(Toeplitz.reshape(1,64), 36, axis=0)

    # Roll each row by one group rolling.
    Toeplitz[np.arange(1,36,6)] = np.roll(Toeplitz[np.arange(1,36,6)], 1)
    Toeplitz[np.arange(2,36,6)] = np.roll(Toeplitz[np.arange(2,36,6)], 2)
    Toeplitz[np.arange(3,36,6)] = np.roll(Toeplitz[np.arange(3,36,6)], 3)
    Toeplitz[np.arange(4,36,6)] = np.roll(Toeplitz[np.arange(4,36,6)], 4)
    Toeplitz[np.arange(5,36,6)] = np.roll(Toeplitz[np.arange(5,36,6)], 5)

    # Roll rach group by 8 each time.
    # Notice that when we roll the whole last rows, we can roll them "stairs by stairs" 
    # rather than roll them by groups.
    Toeplitz[np.arange(6,36)] = np.roll(Toeplitz[np.arange(6,36)], 8)
    Toeplitz[np.arange(12,36)] = np.roll(Toeplitz[np.arange(12,36)], 8)
    Toeplitz[np.arange(18,36)] = np.roll(Toeplitz[np.arange(18,36)], 8)
    Toeplitz[np.arange(24,36)] = np.roll(Toeplitz[np.arange(24,36)], 8)
    Toeplitz[np.arange(30,36)] = np.roll(Toeplitz[np.arange(30,36)], 8)
    
    # Now, we have got the Toeplitz matrix by some tricky rolling.
    # Then we just need to make a matrix multiplication between the 'Toeplitz' and 'img'.
    output = np.matmul(Toeplitz, padding_img)
    
    # Reshape the output to (6*6) because of the zero-padding.
    output.reshape(6, 6)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    