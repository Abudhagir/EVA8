
Objective:

    Run this network Links to an external site..
    Fix the network above:
    
    change the code such that it uses GPU and
    change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
    total RF must be more than 44
    one of the layers must use Depthwise Separable Convolution
    one of the layers must use Dilated Convolution
    use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    use albumentation library and apply:
    horizontal flip
    shiftScaleRotate
    coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset),                mask_fill_value = None)
    achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
    upload to Github
    Attempt S6-Assignment Solution.
    Questions in the Assignment QnA are:
    copy paste your model code from your model.py file (full code) [125]
    copy paste output of torchsummary [125]
    copy-paste the code where you implemented albumentation transformation for all three transformations [125]
    copy paste your training log (you must be running validation/text after each Epoch [125]
    Share the link for your README.md file. [200]


Key Idea: Replacing Maxpooling by strided convolutions(i.e convolution layer with a stride of 2
    `we can think of strided convolutions as learnable pooling`
Replacing maxpoling with convolution layer with stride of 2 will be a little bit more expensive, because we have now more parameters. 

But in that way, we can also simplify the network in terms of having it look simpler, just saying,okay, we only use convolutions, we don't use anything else, if that's desirable. We still need the activation function, but we don't need pooling layers, for example. I will be a bit costly operation but

Model Architecture

Four convolution blocks
first block: | Reciptive Field =13
3 dilated convolution with kernel size=3,stride=1, padding=2,dilation=2

Second Block: | Reciptive Field =33
1 dilated convolution with kernel size=3,stride=2,dilation=2,padding=1
2 depthwise separable convoltion with kernel size=3,dilation=2,stride=1,padding=2

third block: | Reciptive Field = 57
1 dilated convolution with kernel size=3,stride=1,dilation=2,padding=1
2 depthwise separable kernel with kernel_size=3,stride=1,padding=1

Fourth Block: | Reciptive Field = 121
1 dilated convolution with kernel size=3,dilation=2,stride=2,padding=2
1 depthwise separable convoltion with kernel size=3,stride=1,padding=1
1 dilated convoltion with kernel_size=3,dilation=2,stride=1,padding=2
1 Normal Convolution kernel_size=(3,3),stride=(1,1),padding=1

GlobalAveragePooling Layer
Normal Convolution with kernel_size=(3,3),stride=(1,1),padding=
