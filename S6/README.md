
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
