# ImageFusion

## Method

Cut-out from RGB-imgs -> Slide the cut-out through full image and calculate similarity using torch.nn.functional.Conv2d -> Calculate probability using softmax -> Find most similarity position

Cuting one sepecific area from the image,  using correlation 

To evaluate the performance of RGB-IR fusion

## Functions
`correlaion`: read images, cut one specific shape cut-out, use Torch.nn.functional.conv2d to 
