# ImageFusion

The content in three files are same, ipynb file has visualization result, Correlation_fusion_shorter is shorter and easy to read.

Key work: 1. Align images based on moving distance from origin point. 2. Zoom in IR images to match the size of RGB images(640x480), cut embeded images and circular pad reference images. 3. Correlate images. 

Pipeline: align and export images ---> image processing of IR images ---> choose embeded images and reference images ---> image processing of embeded and reference images ---> Correlate IR images and RGB images separately ---> combine IR and RGB correlation result ---> final result.

## Files Structure

![image](https://github.com/3505473356/ImageFusion/blob/main/Files_structure.png)

## Creating dataset
### Images align

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
