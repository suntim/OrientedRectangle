# -*- coding: UTF-8 -*-
#!/usr/bin/env python
"""图片旋转指定逆时针旋转的角度"""
import cv2
from PIL import Image
from math import *



def Rotat1(img_path,angle):
    img = cv2.imread(img_path)
    height,width=img.shape[:2]
    #旋转后的尺寸
    heightNew=int(width*fabs(sin(radians(angle)))+height*fabs(cos(radians(angle))))
    widthNew=int(height*fabs(sin(radians(angle)))+width*fabs(cos(radians(angle))))

    matRotation=cv2.getRotationMatrix2D((width/2,height/2),angle,1)

    matRotation[0,2] +=(widthNew-width)/2  #重点在这步
    matRotation[1,2] +=(heightNew-height)/2  #重点在这步
    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))

    cv2.imshow("img",img)
    cv2.imshow("imgRotation",imgRotation)
    cv2.waitKey(0)
    return imgRotation

def Rotat2(img_path,angle):
    image = Image.open(img_path)
    # 指定逆时针旋转的角度
    iSRotate = image.rotate(angle)
    iSRotate.show()
    return iSRotate

if __name__ == '__main__':
    img_path = r"C:\Users\Desktop\1111\0001_extbcr_detect_rst.jpg"
    save_path = r"C:\Users\Desktop\2222\rot_0001_extbcr_detect_rst.jpg"
    imgRotation = Rotat1(img_path, 45)
    # imgRotation.save(save_path)
    cv2.imwrite(save_path,imgRotation)
