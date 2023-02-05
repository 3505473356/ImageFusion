#!/usr/bin/env python
# coding: utf-8

# ## Files_Structure
# 
# "Root(Category)-------Videos------RGB_images_file
# "                  |"
# "                  |------IR__images_file"

# In[ ]:


import os
import cv2
import numpy as np


# In[ ]:


############################################################################################
# Adjust images. Zoom in IR images, cut embeded images and circularly pad reference images #
############################################################################################

def adjust_imgs(img_list, root_path, isInfrared, isReference=True, cut_area=0.75):
    """
    Processing images, padding reference images and cuting embeded images
    """
    imgs = []
    print("Loading dataset.")
    for img in img_list:
        img = cv2.imread(root_path+ "/" +img)
        print("Image shape: H:{}, W:{}, C:{}".format(img.shape[0], img.shape[1], img.shape[2]), end='\r')
        
        if isInfrared:
            img = img[:,:,1].reshape(img.shape[0], img.shape[1], 1)
            print("Infrared process: H:{}, W:{}, C:{}".format(img.shape[0], img.shape[1], img.shape[2]), end='\r')
            
        if isReference:
            img = np.pad(img, ((0,0), (int(img.shape[1]*cut_area/2),int(img.shape[1]*cut_area/2)), (0,0)), 'wrap')
            print("Reference imgs has padded, shape: {}".format(img.shape), end='\r')
            
        if not isReference:
            edge_L = int(img.shape[1]*(1-cut_area))
            edge_R = img.shape[1]
            img = img[:, edge_L:, :]
            emb_length = img.shape[1]
            print("Embeded imgs has cut, shape: {}".format(img.shape), end='\r')
            
        imgs.append(img)
    
    return imgs

def adjust_dataset(ref_data_file, emb_data_file, isInfrared):
    """
    Loading dataset and process whole dataset
    """
    ref_img_list = os.listdir(ref_data_file)
    emb_img_list = os.listdir(emb_data_file)
    
    used_imgs_number = 0
    
    if len(ref_img_list) <= len(emb_img_list):
        used_imgs_number = len(ref_img_list)
        emb_img_list = emb_img_list[:used_imgs_number]
        
    else:
        used_imgs_number = len(emb_img_list)
        ref_img_list = ref_img_list[:used_imgs_number] 
    
    ref_img_list = adjust_imgs(ref_img_list, ref_data_file, isInfrared)
    emb_img_list = adjust_imgs(emb_img_list, emb_data_file, isInfrared, isReference=False)
    
    print("{} reference images, {} embedded images, {} pair images used.".format(len(ref_img_list), len(emb_img_list), used_imgs_number, end = '/r')) 
    print("**********************************************************************************")
    
    return np.array(ref_img_list), np.array(emb_img_list)


# In[ ]:


######################################################################
# Loading IR and RGB images in Embeded dataset and Reference dataset #
######################################################################

def load_imgs(root):
    RGB_path = os.path.join(root,"RGB")
    IR_path = os.path.join(root,"IR" )
    
    return RGB_path, IR_path

def load_dataset(ref_path, emb_path):
    ref_RGB_path, ref_IR_path = load_imgs(ref_path)
    emb_RGB_path, emb_IR_path = load_imgs(emb_path)
    
    ref_RGB_list, emb_RGB_list = adjust_dataset(ref_RGB_path, emb_RGB_path, isInfrared=False)
    ref_IR_list, emb_IR_list = adjust_dataset(ref_IR_path, emb_IR_path, isInfrared=True)
    
    batch_1, H_ref_RGB, W_ref_RGB, C_ref_RGB= ref_RGB_list.shape
    batch_2, H_ref_IR, W_ref_IR, C_ref_IR= ref_IR_list.shape
    batch_3, H_emb_RGB, W_emb_RGB, C_emb_RGB= emb_RGB_list.shape
    batch_4, H_emb_IR, W_emb_IR, C_emb_IR= emb_IR_list.shape
    
    if batch_1 == batch_2 == batch_3 == batch_4 and H_emb_IR == H_emb_RGB == H_ref_IR == H_ref_RGB:
        batch = batch_1
        H = H_emb_IR
        
    if C_ref_RGB == C_emb_RGB == 3 and C_emb_IR == C_ref_IR == 1:
        C_RGB = 3
        C_IR = 1
    
    if W_emb_IR == W_emb_RGB and W_ref_IR == W_ref_RGB:
        W_emb = W_emb_IR
        W_ref = W_ref_IR
    
    para = [batch, C_RGB, C_IR, H, W_emb, W_ref]
    
    return ref_RGB_list, emb_RGB_list, ref_IR_list, emb_IR_list, para


# In[ ]:


if __name__ == "__main__":
    ref_dataset_path = "path0/2021-09-04-18-17-20/"
    emb_dataset_path = "path0/2021-09-04-18-25-14/"
    
    ref_RGB_list, emb_RGB_list, ref_IR_list, emb_IR_list = load_dataset(ref_dataset_path, emb_dataset_path)

