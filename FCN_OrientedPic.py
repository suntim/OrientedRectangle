                    temp_pos = np.dot(transform, np.array([x, y, 1]).T)
                    transed_x = temp_pos[0] / temp_pos[2]
                    transed_y = temp_pos[1] / temp_pos[2]
                    transfered_pos[x][y] = (int(transed_x), int(transed_y))
            # 生成 一个空的彩色图像
            result_img = np.zeros((show_height, show_width, 3), np.uint8)
            # print result_img.shape

            for x in range(show_width):
                for y in range(show_height):
                    # print "transfered_pos[x][y][1] = ",transfered_pos[x][y][1]
                    result_img[y][x] = proimage[int(transfered_pos[x][y][1])][int(transfered_pos[x][y][0])]

            cv2.imwrite(os.path.join(save_path, "Rotate{}_{}.jpg".format(imgName, i)), result_img)
            cv2.imshow("result_img", result_img)
            cv2.waitKey()



if __name__ == '__main__':
    img_dir = r'D:\Bill\TrainSegBill\test\orig'
    label_dir = r'D:\Bill\TrainSegBill\test\label'
    save_path = r'D:\Bill\TrainSegBill\test'
    for imgName in os.listdir(img_dir):
        if re.match(".*[.]jpg$",imgName):
            img_path = os.path.join(img_dir,imgName)
            label_path = os.path.join(label_dir,imgName.split('.')[0]+'.png')
            assert os.path.exists(label_path),"Not exist {} !".format(label_path)
            SegRoIFunc(img_path, label_path, save_path, Method = 2)

