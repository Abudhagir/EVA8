GOAL:

your colab file must: train resnet18 for 20 epochs on the CIFAR10 dataset show loss curves for test and train datasets show a gallery of 10 misclassified images show gradcam output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. ragecursing_facecursing_facecursing_facecursing_face Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. Train for 20 epochs Get 10 misclassified images Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class) Apply these transforms while training: RandomCrop(32, padding=4) CutOut(16x16)

**syed_eva8** - Folder is created

    models, utils and main files are updated here.
    
    ![image](https://user-images.githubusercontent.com/8513086/218257140-ca4a19fe-a1a0-4a03-944a-8ff6d795d035.png)

Analysis
Attempt 1 - Patience value = 10
Epochs - 40
Best Training Accuracy - 82.46% (39th Epoch)
Best Testing Accuracy - 81.33% (40th Epoch)

Optimizer - Adam (learning rate = 0.01, weight decay = 1e-5
Scheduler - ReduceLROnPlateau (mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
Finding - threshold mode "abs" is necessary for mode = "min" when loss is expected to be negative

Attempt 2 - Patience value = 2
Epochs - 40
Best Training Accuracy - 88.23% (39th Epoch)
Best Testing Accuracy - 84.79% (40th Epoch)

Optimizer - Adam (learning rate = 0.01, weight decay = 1e-5
Scheduler - ReduceLROnPlateau (mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

20 Misclassified Images

![image](https://user-images.githubusercontent.com/8513086/218257153-7ecf40f8-0e39-481f-b7c0-a6755fbeedc7.png)

GRADCAM Images






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
