# ImageFusion

The content in three files are same, ipynb file has visualization result, Correlation_fusion_shorter is shorter and easy to read.

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
