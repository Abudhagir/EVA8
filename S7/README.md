
GOAL:

 
your colab file must:
        train resnet18 for 20 epochs on the CIFAR10 dataset
        show loss curves for test and train datasets
        show a gallery of 10 misclassified images
        show gradcam output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬
    Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 
    Train for 20 epochs
    Get 10 misclassified images
    Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
    Apply these transforms while training:
        RandomCrop(32, padding=4)
        CutOut(16x16)

    **syed_eva8** - Folder is created
        models, utils and main files are updated here.

 PLOT

![image](https://user-images.githubusercontent.com/8513086/218254925-3d468181-c186-4fa1-9e4f-baf3c40fc519.png)

**MISCLASSIFIED IMAGES:**



  **GRADCAM MISCLASSFIED IMAGE**
  
  
![Uploading image.png…]()

