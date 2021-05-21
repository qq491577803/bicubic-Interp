import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

class bicubicInterp():
    def __init__(self,image,scaleRatio,a):
        self.image = image
        self.scaleRatio = scaleRatio
        self.a = a
    def weight(self,x,a):
        x = abs(x)
        if abs(x) < 1:
            w = (a + 2) * (abs(x) ** 3) - (a + 3) * (abs(x) ** 2) + 1
        elif (abs(x) < 2 and abs(x) > 1):
            w = a * (abs(x) ** 3) - 5 * a * (abs(x) ** 2) + 8 * a * abs(x) - 4 * a
        else:
            w = 0
        return w

    def plotWeightKernel(self):
        index = [i / 100 for i in range(-300,300)]
        weight = [self.weight(i,-0.5) for i in index]
        plt.figure("weight kernel")
        plt.plot(index,weight)
        plt.show()

    def padding(self,img):
        H,W,C = img.shape
        pad = np.zeros((H+4,W+4,C))
        pad[2:H+2,2:W+2,:C] = img
        #Pad the first/last two col and row
        pad[2:H+2,0:2,:C]=img[:,0:1,:C]
        pad[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]
        pad[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]
        pad[0:2,2:W+2,:C]=img[0:1,:,:C]
        #Pad the missing eight points
        pad[0:2,0:2,:C]=img[0,0,:C]
        pad[H+2:H+4,0:2,:C]=img[H-1,0,:C]
        pad[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]
        pad[0:2,W+2:W+4,:C]=img[0,W-1,:C]
        return pad

    def interp(self,image):
        self.plotWeightKernel()
        H,W,C = self.image.shape
        dstH = math.floor(H * self.scaleRatio)#908
        dstW = math.floor(W * self.scaleRatio)#1032
        interpImage = np.zeros(shape = (dstH,dstW,C))
        for c in range(C):
            for row in range(dstH):
                for col in range(dstW):
                    x = col / self.scaleRatio + 2
                    y = row / self.scaleRatio + 2

                    dx1 = 1 + x - math.floor(x)
                    dx2 = x - math.floor(x)
                    dx3 = 1 - x + math.floor(x)
                    dx4 = 2 - x + math.floor(x)

                    dy1 = 1 + y - math.floor(y)
                    dy2 = y - math.floor(y)
                    dy3 = 1 - y + math.floor(y)
                    dy4 = 2 - y + math.floor(y)
                    
                    wx1 = self.weight(dx1,self.a)
                    wx2 = self.weight(dx2,self.a)
                    wx3 = self.weight(dx3,self.a)
                    wx4 = self.weight(dx4,self.a)

                    wy1 = self.weight(dy1,self.a)
                    wy2 = self.weight(dy2,self.a)
                    wy3 = self.weight(dy3,self.a)
                    wy4 = self.weight(dy4,self.a)
                    # """
                    # (y-dy1,x-dx1), (y-dy2,x-dx1), (y+dy3,x-dx1), (y+dy4,x-dx1)

                    # (y-dy1,x-dx2), (y-dy2,x-dx2), (y+dy3,x-dx2), (y+dy4,x-dx2)

                    # (y-dy1,x+dx3), (y-dy2,x+dx3), (y+dy3,x+dx3), (y+dy4,x+dx3)

                    # (y-dy1,x+dx4), (y-dy2,x+dx4), (y+dy3,x+dx4), (y+dy4,x+dx4)
                    # """
                    sumx1 = wx1 *  image[int(y-dy1),int(x-dx1),c] + wx2 * image[int(y-dy1),int(x-dx2),c] + wx3 *  image[int(y-dy1),int(x+dx3),c] + wx4 *  image[int(y-dy1),int(x+dx4),c]
                    sumx2 = wx1 *  image[int(y-dy2),int(x-dx1),c] + wx2 * image[int(y-dy2),int(x-dx2),c] + wx3 *  image[int(y-dy2),int(x+dx3),c] + wx4 *  image[int(y-dy2),int(x+dx4),c]
                    sumx3 = wx1 *  image[int(y+dy3),int(x-dx1),c] + wx2 * image[int(y+dy3),int(x-dx2),c] + wx3 *  image[int(y+dy3),int(x+dx3),c] + wx4 *  image[int(y+dy3),int(x+dx4),c]
                    sumx4 = wx1 *  image[int(y+dy4),int(x-dx1),c] + wx2 * image[int(y+dy4),int(x-dx2),c] + wx3 *  image[int(y+dy4),int(x+dx3),c] + wx4 *  image[int(y+dy4),int(x+dx4),c]

                    dst = sumx1 * wy1 + sumx2 * wy2 + sumx3 * wy3 + sumx4 * wy4
                    interpImage[row,col,c] = dst
        return interpImage

    def run(self):
        pad = self.padding(self.image)
        dst = self.interp(pad)
        cv2.imshow("src",self.image)
        cv2.imshow("dst",dst.astype(np.uint8))
        cv2.waitKey(0)

if __name__ == "__main__":
    iPath = r".\image\face.jpeg"
    image = cv2.imread(iPath)
    image = cv2.resize(image,dsize = (200,200))
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    bicubicInterp(image,2,-0.5).run()
