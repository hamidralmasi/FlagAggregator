import numpy as np
import math
import pylab
from PIL import Image
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
class Class_EM:
    def __init__(self):
        pass
 
    def ICV_degreeToRadian(self, degree):
        '''
        Convert degree to radian
        :param degree: degree
        :return: radian
        '''
        return (degree * np.pi) / 180.0
 
 
    def ICV_getImageMidPoints(self, image):
        '''
        This method returns image midpoints
        :param image:
        :return: x_mid, y_mid
        '''
        height, width, num_channels = image.shape
        y_mid = height / 2
        x_mid = width / 2
        return x_mid, y_mid
 
    '''
    rotate image in python
    '''
 
    
    def Class_EMBasedOnDegree(self, image, degree):
        '''
        Rotate image based on degree
        :param image: actual image
        :param degree: degree
        :return: rotated image
        '''
    
        x_mid, y_mid = self.ICV_getImageMidPoints(image)
 
        newimage = self.ICV_canvasSize(image)
 
        newimage_x_mid, newimage_y_mid = self.ICV_getImageMidPoints(newimage)

       
        #### Change #1
        #{

        #for Lotka Volterra
        t = np.linspace(0, degree, 10)
        
        #for rotation
        # t = np.linspace(0, degree, degree+1)
        
        #### Change #1
        #}

        alpha = 2/3
        beta = 4/3

        gamma = 1
        delta = 1

        def lotka(t,coordinates, alpha, beta, gamma, delta):

            x, y = coordinates
            derivate = [(alpha * x) - (beta * x * y), (delta * x * y) - (gamma * y)]
            return derivate
            
        def rotate( t,coordinates, alpha, beta, gamma, delta):

            x, y = coordinates
            derivate = [-y, x]
            return derivate
        
        

        for img_x in range(image.shape[0]):
            for img_y in range(image.shape[1]):
                # This will start with midpoints in first iteration
                # and then propagates back
                y_prime = y_mid - img_y
                x_prime = x_mid - img_x

                #### Change #2
                #{
                #for Lotka Volterra
                sol = solve_ivp(lotka,  [0,t[-1]], [x_prime, y_prime], method='LSODA', t_eval = t, args=(alpha, beta, gamma, delta))
                
                #for rotation
                # sol = solve_ivp(rotate, [0,t[-1]], [x_prime, y_prime], method='LSODA', t_eval = t, args=(alpha, beta, gamma, delta))
                
                #### Change #2
                #}

                #takes the last x,y points of the trajectory
                x_new = sol.y[0][-1]
                y_new = sol.y[1][-1]


                # Adjust each new points wrt newImage mid points
                xdist = int(newimage_x_mid - x_new)
                ydist = int(newimage_y_mid - y_new)
                if (xdist >=0) and (ydist >=0) and (xdist < image.shape[0]) and ydist < image.shape[1]:
                    newimage[img_x][img_y][:] = image[xdist][ydist][:]
 
 
        return newimage
 
 
    def ICV_canvasSize(self, image):
        '''
        This method generates Canvas Size based on Input Image
        :param image: image array
        :return: canvas
        '''
        height, width, num_channels = image.shape
        # max_len = int(math.sqrt(width * width + height * height))
        canvas = np.zeros((height, width, 3))
        return canvas
 
 
    def ICV_plotImage(self, newImage, filename):
        '''
        This method plots the new image
        :param newImage:
        :return: Null
        '''
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
 
        output_image = Image.fromarray(newImage.astype("uint8"))
        plt.imshow(output_image)
        plt.savefig(filename,bbox_inches='tight', pad_inches=0)
        plt.show()
 
 
def main():
    image = Image.open('image.jpg')
    image = np.asarray(image)
    # print(image.shape)
    rotation = Class_EM()

    #### Change #3
    #{

    #for Lotka Volterra
    angles = [0.001,0.002,0.003,0.004,0.005, 0.006, 0.007, 0.008, 0.009, 0.010]

    #for Rotation
    # angles = [30, 45, 60, 90, 135, 180, 225, 270, 405, 540, 630, 720]
    
    #### Change #3
    #}

    for angle in angles:
    
        newImage1 = rotation.Class_EMBasedOnDegree(image, angle)
        filename = "image" + str(angle) + "_.jpg" 
        rotation.ICV_plotImage(newImage1,filename)
 

if __name__ == "__main__":
    main()
