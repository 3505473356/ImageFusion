#! /usr/bin/env python

import rospy
import rosbag
import cv2
import numpy as np
import os
import pandas as pd
from cv_bridge import CvBridge
import sys

def cut_roi(fx_IR, fy_IR, cx_IR, cy_IR, fx_RGB, fy_RGB, cx_RGB, cy_RGB, width, height):

    roi_width = width * fx_IR / fx_RGB
    roi_height = height * fy_IR / fy_RGB

    shift_x = cx_IR - cx_RGB
    shift_y = cy_IR - cy_RGB

    roi_L = (width-roi_width)/2 - shift_x
    roi_R = (width+roi_width)/2 - shift_x

    roi_U = (height-roi_height)/2 - shift_y
    roi_D = (height+roi_height)/2 - shift_y

    return roi_L, roi_R, roi_U, roi_D

def get_time_list(RGB_msgs, IR_msgs):
    RGB_time_list = []
    IR_time_list = []
    
    for RGB_msg, IR_msg in zip(RGB_msgs, IR_msgs):

        RGB_time_list.append(RGB_msg.timestamp.to_sec())
        IR_time_list.append(IR_msg.timestamp.to_sec())

    rospy.loginfo("Time lists of RGB(%s) and IR images(%s) are token", len(RGB_time_list), len(IR_time_list))
    rospy.loginfo("\n")

    # RGB_pd = pd.DataFrame(np.array(RGB_time_list))
    # IR_pd = pd.DataFrame(np.array(IR_time_list))

    # writer_RGB = pd.ExcelWriter('RGB.xlsx')
    # writer_IR = pd.ExcelWriter('IR.xlsx')

    # RGB_pd.to_excel(writer_RGB, 'page_1')
    # IR_pd.to_excel(writer_IR, 'page_1')

    # writer_RGB.save()
    # writer_IR.save()

    return RGB_time_list, IR_time_list

def parse_cams_info(RGB_cam_msgs, IR_cam_msgs, count=False, flag=0, end=3):

    fx_RGB = 0
    fx_IR = 0
    cx_RGB = 0
    cx_IR = 0

    fy_RGB = 0
    fy_IR = 0
    cy_RGB = 0
    cy_IR = 0

    for RGB_cam_msg, IR_cam_msg in zip(RGB_cam_msgs, IR_cam_msgs):

        if count and flag > end:
            break
        
        else:
            width_RGB, height_RGB = RGB_cam_msg.message.width, RGB_cam_msg.message.height
            width_IR, height_IR = IR_cam_msg.message.width, IR_cam_msg.message.height

            K_RGB, K_IR = np.array(RGB_cam_msg.message.K).reshape(3,3), np.array(IR_cam_msg.message.K).reshape(3,3)

            if fx_RGB != K_RGB[0][0] or fy_RGB != K_RGB[1][1] or cx_RGB != K_RGB[0][2] or cy_RGB != K_RGB[1][2]:
                if fx_RGB == fy_RGB == cx_RGB == cy_RGB == 0:
                    fx_RGB, fy_RGB, cx_RGB, cy_RGB = K_RGB[0][0], K_RGB[1][1], K_RGB[0][2], K_RGB[1][2]
                    rospy.loginfo("\033[32mLoad intrinsic parameters.  \033[0m")
                    rospy.loginfo("fx: %s, fy: %s, cx: %s, cy: %s", fx_RGB, fy_RGB, cx_RGB, cy_RGB)

                else:
                    rospy.logwarn("Some parameters in RGB intrinsic matrix changed!")
                    break

            if fx_IR != K_IR[0][0] or fy_IR != K_IR[1][1] or cx_IR != K_IR[0][2] or cy_IR != K_IR[1][2]:
                if fx_IR == fy_IR == cx_IR == cy_IR == 0:
                    fx_IR, fy_IR, cx_IR, cy_IR = K_IR[0][0], K_IR[1][1], K_IR[0][2], K_IR[1][2]
                    rospy.loginfo("\033[32mLoad intrinsic parameters.  \033[0m")
                    rospy.loginfo("fx: %s, fy: %s, cx: %s, cy: %s", fx_IR, fy_IR, cx_IR, cy_IR)
                    rospy.loginfo("\n")
                else:
                    rospy.logwarn("Some parameters in IR intrinsic matrix changed!")
                    break
            
            flag += 1

    flag = 0

    if height_IR == height_RGB and width_IR == width_RGB:
        width = width_IR
        height = height_IR

        roi_L, roi_R, roi_U, roi_D = cut_roi(fx_IR, fy_IR, cx_IR, cy_IR, fx_RGB, fy_RGB, cx_RGB, cy_RGB, width, height)

        rospy.loginfo("\033[32m Information  \033[0m")
        rospy.loginfo("IR-camera: fx: %s, fy: %s, cx: %s, cy: %s", fx_IR, fy_IR, cx_IR, cy_IR)
        rospy.loginfo("RGB-camera: fx: %s, fy: %s, cx: %s, cy: %s", fx_RGB, fy_RGB, cx_RGB, cy_RGB)
        rospy.loginfo("Image: width: %s, height: %s", width, height)
        rospy.loginfo("ROI in IR image: left boundary: %s, right boundary: %s, top boundary: %s, down boundary: %s", roi_L, roi_R, roi_U, roi_D)
        rospy.loginfo("\033[32m *******************End******************* \033[0m")

    else:
        rospy.logwarn("The resolution of IR-RGB pair images not same")

    return roi_L, roi_R, roi_U, roi_D, width, height

def align_img_pair(RGB_img_topic, IR_img_topic, status_msgs, roi_L, roi_R, roi_U, roi_D, width, height, count=False):
    # extract imgs from IR_cam_msgs and RGB_cam_msgs
    i = 0

    distance = []
    status_time_list = []

    RGB_img_list = []
    IR_img_list = []

    bridge = CvBridge()

    RGB_img_msgs = bag.read_messages(RGB_img_topic)
    IR_img_msgs = bag.read_messages(IR_img_topic)

    RGB_time_list, IR_time_list = get_time_list(RGB_img_msgs, IR_img_msgs)

    rospy.loginfo("RGB_time %s, IR_time %s", len(RGB_time_list), len(IR_time_list))
    
    rospy.loginfo("Loading odom distance and corresponding time")
    for status in status_msgs:
        i += 1
        distance.append(status.message.currentDistanceMM)
        rospy.loginfo("sxxxxxxxx, %s", status)
        if len(distance) > 100:
            if (distance[-1] - distance[-100]) > 6400:
                differ = distance[-1] - distance[-100]
                status_time_list.append(status.timestamp.to_sec())
                # rospy.loginfo("(%s) %s message (%s), diff: %s - %s = %s ",len(distance), i, len(status_time_list), distance[-1], distance[-100], differ)
        else:
            continue

    rospy.loginfo("Odom distance and corresponding time (%s) stored", len(status_time_list))

    index = 0
    count_RGB = 0
    count_IR = 0

    RGB_img_msgs = bag.read_messages(RGB_img_topic)
    IR_img_msgs = bag.read_messages(IR_img_topic)
    
    for RGB_msg, IR_msg in zip(RGB_img_msgs, IR_img_msgs):

        if index+1 == len(RGB_time_list) or count_IR == count_RGB == len(status_time_list)-1:
            rospy.loginfo("Finished")
            break

        if count_RGB < len(status_time_list):
            if RGB_time_list[index] <= status_time_list[count_RGB] and RGB_time_list[index+1] > status_time_list[count_RGB]:
                RGB_img = bridge.imgmsg_to_cv2(RGB_msg.message, "bgr8")
                RGB_img_list.append(RGB_img)
                count_RGB += 1
                # rospy.loginfo("%s RGB-Image is saved, status_time_list: %s", index, count_RGB)

        if count_IR < len(status_time_list):
            if IR_time_list[index] <= status_time_list[count_IR] and IR_time_list[index+1] > status_time_list[count_IR]:

                # rospy.logwarn("%s, %s, %s, index: %s", IR_time_list[index], status_time_list[count_IR], IR_time_list[index+1], index)

                IR_img = bridge.imgmsg_to_cv2(IR_msg.message, "bgr8")
                IR_img = IR_img[:,:,1].reshape(height,width,1)
                IR_img = IR_img[round(roi_U)-3 : round(roi_D)-3, round(roi_L)-1 : round(roi_R)-1, :]
                IR_img = cv2.resize(IR_img, (width, height))
                IR_img = IR_img.reshape(height,width, 1)
                IR_img_list.append(IR_img)

                count_IR += 1
                # rospy.loginfo("%s IR-Image is saved, status_time_list: %s", index, count_IR)

        # rospy.loginfo("%s, %s, %s, index: %s", IR_time_list[index], status_time_list[count_IR], IR_time_list[index+1], index)

        index += 1
        # rospy.loginfo("\n")
    rospy.loginfo("%s, %s, %s, %s, %s, %s", index, type(status_time_list[count_IR]), type(IR_time_list[index]), len(RGB_time_list), len(IR_time_list), len(status_time_list))
    rospy.loginfo("%s RGB images stored, %s IR images stored", len(RGB_img_list), len(IR_img_list))

    # cv2.imshow("RGB_img", RGB_img)
    # cv2.imshow("IR_img", IR_img)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    # align images
    return RGB_img_list, IR_img_list

def get_bag_list(dir):
    bag_list = []

    if os.path.isfile(dir):
        bag_list.append(dir)

    elif os.path.isdir(dir):
        for bag in os.listdir(dir):
            bag_list.append(bag)

    return dir, bag_list

def check_dataset(RGB_img_list, IR_img_list):
    if len(RGB_img_list) != len(IR_img_list):
        with open("Broken_rosbag.txt", "a") as f:
            f.write(str(bag_path) + "\n")
            f.close()
            
        return 0
    
    else:
        return True

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        rospy.loginfo("%s is created", path)
    
    else:
        rospy.loginfo("%s exists", path)

    return path

def store_dataset(root, bag_path, RGB_img_list, IR_img_list, flag=False):
    count = 1
    name = root + "/" + bag_path.split(".")[0]
    dataset_path = mkdir(name)
    dataset_path_RGB = mkdir(dataset_path + "/RGB")
    dataset_path_IR = mkdir(dataset_path + "/IR")

    for rgb_img, ir_img in zip(RGB_img_list, IR_img_list):
        if count <= len(RGB_img_list):
            rgb_img = np.array(rgb_img)
            ir_img = np.array(ir_img)
            cv2.imwrite(dataset_path_RGB + "/rgb_" + str(count) +".png", rgb_img)
            cv2.imwrite(dataset_path_IR + "/ir_" + str(count) +".png", ir_img)
            
        if count > len(RGB_img_list):
            rospy.loginfo("%s pair images stored.", count-1)

        if flag and count == int(len(RGB_img_list)/4):
            break

        count += 1


if __name__ == "__main__":

    bags_file = "/home/u2004/Desktop/path_1"

    root, bag_list = get_bag_list(bags_file)

    status_topic = "/status"
    RGB_cam_topic = "/camera/color/camera_info"
    IR_cam_topic = "/camera/infra1/camera_info" 
    RGB_img_topic = "/camera/color/image_raw"
    IR_img_topic = "/camera/infra1/image_rect_raw"
    
    rospy.init_node("read_bag")

    rospy.loginfo("Root path is: %s, found %s rosbags", root, len(bag_list))

    for bag_path in bag_list:
        bag = rosbag.Bag(root + "/" + bag_path, "r")

        rospy.loginfo("Folder %s is created.", str(bag_path.split(".")[0]))

        status_msg = bag.read_messages(status_topic)
        RGB_cam_info = bag.read_messages(RGB_cam_topic)
        IR_cam_info = bag.read_messages(IR_cam_topic)

        roi_L, roi_R, roi_U, roi_D, width, height = parse_cams_info(RGB_cam_info, IR_cam_info, count=False)

        RGB_img_list, IR_img_list = align_img_pair(RGB_img_topic, IR_img_topic, status_msg, roi_L, roi_R, roi_U, roi_D, width, height, count=False)

        check_result = check_dataset(RGB_img_list, IR_img_list)

        if check_result:
            store_dataset(root, bag_path, RGB_img_list, IR_img_list)
        
        else:
            rospy.logwarn("Rosbag %s is broken, skip it.", str(bag_path))
            continue

        bag.close() 