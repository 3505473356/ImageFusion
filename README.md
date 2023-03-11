# ImageFusion

## Introudction

This project aims to improve Visual teach and repeat navigation (VT&R) systems' performance in illumination changes using one Siamese-CNN trained in a contrastive learning manner which fuses infrared and RGB images at the decision and feature levels. It is based on the pervious work in (VT&R) system which aims to find the horizontal displacement between prerecorded and currently perceived images required to steer a robot towards the previously traversed path<sup>[1]</sup>.

## Preparation

`Build dataset`: According to wheel odometry, IR and RGB image pairs are aligned and extracted at a regular distance in rosbags. Rectify these image pairs and store them as dataset. Below is one example:

 <img src="https://github.com/3505473356/ImageFusion/blob/main/Picture/Align_images.png" width = "600" height = "500" alt="example" align=center />

### Content
path0: 11 videos and 18,490 images(Including IR and RGB images)
path1: 7 videos and 17700 images(Including IR and RGB images)
path2: 4 videos and 9028 images(Including IR and RGB images)


<!-- ## Dataset
### Files Structure
![image](https://github.com/3505473356/ImageFusion/blob/main/Picture/Files_structure.png) -->

## Method
### Decision level

1. Single image pipeline

Cut out embedded images and pad reference images -> Slide the embedded images through reference images and calculate similarity using torch.nn.functional.Conv2d -> Normalization -> Calculate likelihood using softmax -> Find most likely position -> Evaluate the distance between ground truth and given output by Absolute Error.

2. Fusion image pipeline

Take same embedded RGB and IR images -> single image processing seperately -> dot multiply two likelihood arrays -> Find most likely position -> Evaluate the distance between ground truth and given output by Absolute Error.

### Feature level
1. Combine infrared image and RGB image into one four channel image (Red, Green, Blue, Infrared).

2. Train one Siamse-CNN for combined imgae input.

### Evaluation

1. Absoulte error: Output displacement - ground truth
2. Standard deviation of AE.

## Result
At the decision level, the improvement is unnoticeable, but the fusion results are obust to extreme environmental changes. Experiment result.

 <img src="https://github.com/3505473356/ImageFusion/blob/main/Picture/Result_path0.png" width = "600" height = "500" alt="example" align=center />
 
One detailed result shows below:

 <img src="https://github.com/3505473356/ImageFusion/blob/main/Picture/example.png" width = "600" height = "500" alt="example" align=center />
