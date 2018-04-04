# -*- coding: UTF-8 -*-
#!/usr/bin/env python
""""
使用的xml格式：
 <bndbox>四个点
      <X>5795</X>
      <Y>4330</Y>
      <X>1488</X>
      <Y>1247</Y>
      <X>1622</X>
      <Y>1044</Y>
      <X>6248</X>
      <Y>4298</Y>
    </bndbox>
"""
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import re
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体

def plotXML(xml_path,img_dir):
    tree = ET.parse(xml_path)
    root_name = tree.getroot()
    # 遍历xml文档
    for childs in root_name:
        for fName in childs.iter('filename'):
            imgName = fName.text+".jpg"
            print os.path.join(img_dir,imgName)
            img = cv2.imread(os.path.join(img_dir,imgName))
            h,w = img.shape[:2]
            # print h,w
            # reImg = cv2.resize(img,(int(w/20),int(h/20)))#这里是w，h
            # cv2.imwrite(os.path.join(save_dir,"123.jpg"),reImg)
            # cv2.imshow(imgName,img)
            # cv2.waitKey(0)
            # cv2.imshow("img",img)
            # cv2.waitKey(0)
            img = img[:, :, (2, 1, 0)]
            img_copy = img.copy()
        for chil in childs.iter('object'):
            for name in chil.iter('name'):
                class_name = name.text
                print class_name
                for x1,y1,x2,y2,x3,y3,x4,y4 in chil.iter('bndbox'):
                    # print x1.text,y1.text,x2.text,y2.text,x3.text,y3.text,x4.text,y4.text
                    x1_v = int(x1.text)
                    y1_v = int(y1.text)
                    x2_v = int(x2.text)
                    y2_v = int(y2.text)
                    x3_v = int(x3.text)
                    y3_v = int(y3.text)
                    x4_v = int(x4.text)
                    y4_v = int(y4.text)
                    print x1_v,y1_v,x2_v,y2_v,x3_v,y3_v,x4_v,y4_v

                    # fig, ax = plt.subplots(figsize=(12, 12))
                    # ax.imshow(img_copy, aspect='equal')

                    cv2.line(img_copy, (x1_v, y1_v), (x2_v, y2_v), (15,255,15),2)  # cv2.line(img,pt1,pt2,color,thickness,lineType,shift)
                    cv2.line(img_copy, (x2_v, y2_v), (x3_v, y3_v), (15,255,15),2)  # cv2.line(img,pt1,pt2,color,thickness,lineType,shift)
                    cv2.line(img_copy, (x3_v, y3_v), (x4_v, y4_v), (15,255,15),2)  # cv2.line(img,pt1,pt2,color,thickness,lineType,shift)
                    cv2.line(img_copy, (x4_v, y4_v), (x1_v, y1_v), (15,255,15),2)  # cv2.line(img,pt1,pt2,color,thickness,lineType,shift)
                    # cv2.floodFill()
                    cv2.putText(img_copy, class_name , (x1_v, y1_v-2), font, 0.9, (0, 255, 255), 2)
                    # ax.text(x2_v, y2_v - 2, '{:s}'.format(class_name), bbox=dict(facecolor='m', alpha=0.5),
                    #         fontsize=14, color='black')
                    # cv2.imshow(imgName,img_copy)
                    # cv2.waitKey(0)
    return img_copy



if __name__ == '__main__':
    xml_dir = unicode(r'D:\1', 'utf-8')
    img_dir = unicode(r'D:\1', 'utf-8')
    save_dir = r'C:\Users\Desktop\1111'
    for fileName in os.listdir(xml_dir):
        if re.match(".*.xml$",fileName):
            print("fileName = %s"%fileName)
            xml_path = os.path.join(xml_dir,fileName)
            img_copy = plotXML(xml_path,img_dir)
            output_dir = os.path.join(save_dir, fileName.split('.')[0] + "_detect_rst.jpg")
            cv2.imwrite(output_dir, img_copy)
