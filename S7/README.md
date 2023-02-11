
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
        


**MISCLASSIFIED IMAGES:**

![image](https://user-images.githubusercontent.com/8513086/218254376-faf96305-11eb-4096-8ae3-413f2d8fc55a.png)


  **GRADCAM MISCLASSFIED IMAGE**
  
  ![image](https://user-images.githubusercontent.com/8513086/218254613-33c1baa7-d49e-43f2-9f12-0fbed3a5f5be.png)

  ![image](https://user-images.githubusercontent.com/8513086/218254622-32c2277d-5a41-42ae-b0d4-776a8f972c7a.png)

![image](https://user-images.githubusercontent.com/8513086/218254648-6bb2904e-761e-4e92-b4eb-72873b905445.png)
![image](https://user-images.githubusercontent.com/8513086/218254658-35522bd2-6b6a-4337-9780-7e6cc7cfc25d.png)
