GOAL:

A. You are going to follow the same structure for your Code from now on. So Create:

   1. models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class
   2. main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):
        training and test loops
        data split between test and train
        epochs
        batch size
        which optimizer to run
        do we run a scheduler?
    3. utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:
        image transforms,
        gradcam,
        misclassification code,
        tensorboard related stuff
        advanced training policies, etc
        etc
    4. Name this main repos something, and don't call it Assignment 7. This is what you'll import for all the rest of the assignments. Add a proper readme describing all the files. 
    
B. Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
    pull your Github code to google colab (don't copy-paste code)
    prove that you are following the above structure
    that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
    your colab file must:
        train resnet18 for 20 epochs on the CIFAR10 dataset
        show loss curves for test and train datasets
        show a gallery of 10 misclassified images
        show gradcam 

    Links to an external site. output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬

Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 
Train for 20 epochs
Get 10 misclassified images
Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
Apply these transforms while training:

    RandomCrop(32, padding=4)
    CutOut(16x16)


1. Separate folder is created for models, utils etc (syed_eva8)

    

    models, utils and main files are updated here.
      
      Abudhagir/syed_eva8-  New Repo name
      
        |
        |---------models
        |             |
        |             |------resnet.py       - which consists of code for resnet model creation
        |
        |---------utils
        |             |------augmentation.py    - albumnetation and data augmentation
        |             |------data_handling.py
        |             |------grandcam.py
        |             |------test.py
        |             |------train.py
        |             |------lr_range_test.py
        |             |------fast_lr_range_test.py  - 
        |
        |
        |---------main.py - Which consists of trainng and testin loops, data split, epochs, batch size, optimizer, scheduler
        
    
    
    **LOSS CURVE**

![image](https://user-images.githubusercontent.com/8513086/218257140-ca4a19fe-a1a0-4a03-944a-8ff6d795d035.png)


Analysis

Epochs - 20
Best Training Accuracy - 87.77% (20th Epoch)
Best Testing Accuracy - 85.34% (19th Epoch)

Optimizer - Adam (learning rate = 0.01, weight decay = 1e-5
Scheduler - ReduceLROnPlateau (mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
Finding - threshold mode "abs" is necessary for mode = "min" when loss is expected to be negative

Attempt 2 - Patience value = 2
Epochs - 20
Best Training Accuracy - 88.23% (39th Epoch)
Best Testing Accuracy - 84.79% (40th Epoch)

Optimizer - Adam (learning rate = 0.01, weight decay = 1e-5
Scheduler - ReduceLROnPlateau (mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

20 Misclassified Images

![image](https://user-images.githubusercontent.com/8513086/218257153-7ecf40f8-0e39-481f-b7c0-a6755fbeedc7.png)

GRADCAM Images - 20 Images

![Grad_images](https://user-images.githubusercontent.com/8513086/218419991-1e1943e2-8478-422e-977b-82212cf74842.png)
![Grad_images2](https://user-images.githubusercontent.com/8513086/218420099-3693c71a-0c8c-45d9-a8a6-722809aba262.png)


Log
EPOCH: 1

Loss=1.4999511241912842 Batch_id=390 LR=0.01000 Accuracy=33.22: 100%|██████████| 391/391 [00:47<00:00,  8.27it/s]

Test set: Average loss: 0.0120, Accuracy: 4497/10000 (44.97%)

EPOCH: 2

Loss=0.9853852391242981 Batch_id=390 LR=0.01000 Accuracy=51.06: 100%|██████████| 391/391 [00:46<00:00,  8.45it/s]

Test set: Average loss: 0.0093, Accuracy: 5847/10000 (58.47%)

EPOCH: 3

Loss=0.9823883175849915 Batch_id=390 LR=0.01000 Accuracy=61.92: 100%|██████████| 391/391 [00:46<00:00,  8.42it/s]

Test set: Average loss: 0.0085, Accuracy: 6176/10000 (61.76%)

EPOCH: 4

Loss=0.740376889705658 Batch_id=390 LR=0.01000 Accuracy=70.42: 100%|██████████| 391/391 [00:46<00:00,  8.46it/s]

Test set: Average loss: 0.0065, Accuracy: 7127/10000 (71.27%)

EPOCH: 5

Loss=0.6558500528335571 Batch_id=390 LR=0.01000 Accuracy=75.03: 100%|██████████| 391/391 [00:46<00:00,  8.40it/s]

Test set: Average loss: 0.0062, Accuracy: 7403/10000 (74.03%)

EPOCH: 6

Loss=0.5912505388259888 Batch_id=390 LR=0.01000 Accuracy=77.85: 100%|██████████| 391/391 [00:46<00:00,  8.44it/s]

Test set: Average loss: 0.0048, Accuracy: 7911/10000 (79.11%)

EPOCH: 7

Loss=0.45842161774635315 Batch_id=390 LR=0.01000 Accuracy=79.95: 100%|██████████| 391/391 [00:46<00:00,  8.49it/s]

Test set: Average loss: 0.0047, Accuracy: 7988/10000 (79.88%)

EPOCH: 8

Loss=0.4735172390937805 Batch_id=390 LR=0.01000 Accuracy=81.23: 100%|██████████| 391/391 [00:45<00:00,  8.52it/s]

Test set: Average loss: 0.0047, Accuracy: 7961/10000 (79.61%)

EPOCH: 9

Loss=0.45328935980796814 Batch_id=390 LR=0.01000 Accuracy=82.39: 100%|██████████| 391/391 [00:46<00:00,  8.48it/s]

Test set: Average loss: 0.0050, Accuracy: 7869/10000 (78.69%)

EPOCH: 10

Loss=0.42412853240966797 Batch_id=390 LR=0.01000 Accuracy=83.41: 100%|██████████| 391/391 [00:45<00:00,  8.52it/s]

Test set: Average loss: 0.0043, Accuracy: 8120/10000 (81.20%)

EPOCH: 11

Loss=0.5910761952400208 Batch_id=390 LR=0.01000 Accuracy=84.01: 100%|██████████| 391/391 [00:45<00:00,  8.51it/s]

Test set: Average loss: 0.0047, Accuracy: 8077/10000 (80.77%)

EPOCH: 12

Loss=0.35608142614364624 Batch_id=390 LR=0.01000 Accuracy=84.66: 100%|██████████| 391/391 [00:45<00:00,  8.60it/s]

Test set: Average loss: 0.0042, Accuracy: 8202/10000 (82.02%)

EPOCH: 13

Loss=0.4100375771522522 Batch_id=390 LR=0.01000 Accuracy=85.23: 100%|██████████| 391/391 [00:45<00:00,  8.57it/s]

Test set: Average loss: 0.0039, Accuracy: 8353/10000 (83.53%)

EPOCH: 14

Loss=0.49094828963279724 Batch_id=390 LR=0.01000 Accuracy=85.82: 100%|██████████| 391/391 [00:45<00:00,  8.54it/s]

Test set: Average loss: 0.0038, Accuracy: 8402/10000 (84.02%)

EPOCH: 15

Loss=0.3284186124801636 Batch_id=390 LR=0.01000 Accuracy=86.15: 100%|██████████| 391/391 [00:45<00:00,  8.60it/s]

Test set: Average loss: 0.0039, Accuracy: 8370/10000 (83.70%)

EPOCH: 16

Loss=0.44286054372787476 Batch_id=390 LR=0.01000 Accuracy=86.89: 100%|██████████| 391/391 [00:45<00:00,  8.62it/s]

Test set: Average loss: 0.0042, Accuracy: 8292/10000 (82.92%)

EPOCH: 17

Loss=0.465336412191391 Batch_id=390 LR=0.01000 Accuracy=87.10: 100%|██████████| 391/391 [00:45<00:00,  8.60it/s]

Test set: Average loss: 0.0037, Accuracy: 8462/10000 (84.62%)

EPOCH: 18

Loss=0.4775037169456482 Batch_id=390 LR=0.01000 Accuracy=87.12: 100%|██████████| 391/391 [00:45<00:00,  8.63it/s]

Test set: Average loss: 0.0036, Accuracy: 8515/10000 (85.15%)

EPOCH: 19

Loss=0.3650830090045929 Batch_id=390 LR=0.01000 Accuracy=87.45: 100%|██████████| 391/391 [00:45<00:00,  8.65it/s]

Test set: Average loss: 0.0034, Accuracy: 8534/10000 (85.34%)

EPOCH: 20

Loss=0.3687034249305725 Batch_id=390 LR=0.01000 Accuracy=87.77: 100%|██████████| 391/391 [00:45<00:00,  8.67it/s]

Test set: Average loss: 0.0038, Accuracy: 8453/10000 (84.53%)
