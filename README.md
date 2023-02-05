# ImageFusion

Keywork: 1. Align images. 2. Image process. 3. Correlate images. 

Pipeline: 1. Align images: Exract images in the same moving distance using sensors' data in rosbags. Then export IR images and RGB images separately. \\
2. Image processing: Rectify IR and RGB images into the same shape (640x480). Define reference dataset and embeded datset, then cut embeded images and circular pad reference images.

![image](https://github.com/3505473356/ImageFusion/blob/main/Align_images.png)

3. Correlation: Correlate IR reference images and IR embeded images and output the IR likelihood map. Same for RGB images and output the RGB likelihood map. Combine IR and RGB likelihood maps, then output fusion map which is the basis of finding displacement of embeded images.

![image](https://github.com/3505473356/ImageFusion/blob/main/Correlate_result.png)

## Files Structure

![image](https://github.com/3505473356/ImageFusion/blob/main/Files_structure.png)

## Dataset

### Problems

## Image Fusion

## Method

1. Single image processing

Cut-out from imgs -> Slide the cut-out through full image and calculate similarity using torch.nn.functional.Conv2d -> Calculate probability using softmax -> Find most likely position -> Evaluate the distance between ground truth and given output by Absolute Error.

2. Fusion image processing

Take same cut-out in RGB and IR imgs -> single image processing seperately -> dot multiply two probability arrays -> Find most likely position -> Evaluate the distance between ground truth and given output by Absolute Error.

3. Evaluate performance

Collecting three AE arrarys including all cut-outs in single RGB-img, single IR-img and fusion img respectively -> Compare MAE(Mean AE) of three AE arrays and compare them -> Repeat it for more images and compare the mean MAE in all three types' correlation.

## Functions
`correlaion`: read images, cut one specific shape cut-out, use Torch.nn.functional.conv2d to generate similarity array and use softmax calculate probability array, output most likely position, evaluate perfomance and show result. 

`log_fusion`: dot multiply RGB and IR probability arrays, output most likely position, evaluate perfomance and show result

`abs_fusion`: dot multiply RGB and IR similarity arrays and use softmax deal with fused similarity array, generate probability array, evaluate performance and show result. (This method is not good)
