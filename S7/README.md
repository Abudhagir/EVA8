GOAL:

your colab file must: train resnet18 for 20 epochs on the CIFAR10 dataset show loss curves for test and train datasets show a gallery of 10 misclassified images show gradcam output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. ragecursing_facecursing_facecursing_facecursing_face Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. Train for 20 epochs Get 10 misclassified images Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class) Apply these transforms while training: RandomCrop(32, padding=4) CutOut(16x16)

**syed_eva8** - Folder is created
    models, utils and main files are updated here.
    
    ![image](https://user-images.githubusercontent.com/8513086/218257140-ca4a19fe-a1a0-4a03-944a-8ff6d795d035.png)
20 Misclassified Images

![image](https://user-images.githubusercontent.com/8513086/218257153-7ecf40f8-0e39-481f-b7c0-a6755fbeedc7.png)

GRADCAM Images

