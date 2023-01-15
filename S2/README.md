![image](https://user-images.githubusercontent.com/8513086/212534910-8c7e1b69-4c5d-4d86-ac37-ee6299add979.png)



Data generation strategy (basically the class/method you are using for random number generation)
def __getitem__(self, index):
    image = self.MNISTDataset[index][0]
    label = self.MNISTDataset[index][1]
    randomNo = random.randint(0,9)

    #Create one hot encoding for random number 
    one_hotrandomNo = torch.nn.functional.one_hot(torch.arange(0, 10))

    #add actual label and random number
    sum = label + randomNo
    return image, label, one_hotrandomNo[randomNo], sum


How you have combined the two inputs (basically which layer you are combining)
    #concatenate second input to the output from above convolution
    x1 = torch.cat((x, randomNumber), dim=1)
        
How you are evaluating your results 

        # compute the loss occured
        mnist_loss = F.nll_loss(output, target)
        addition_loss = F.nll_loss(sum_output, sum)
        loss= (mnist_loss + addition_loss)/2

        epoch_loss += loss.item()
        
        
must mention what loss function you picked and why!

Negative Loglihood Loss Function

Logs:
Epoch 1 : 
Train set: Average loss: 1.3457
Val set: Average loss: 1.309, MNist Accuracy:95.7, Sum_Accuracy:13.82

Epoch 2 : 
Train set: Average loss: 1.2343
Val set: Average loss: 1.220, MNist Accuracy:97.08, Sum_Accuracy:18.54

Epoch 3 : 
Train set: Average loss: 1.2543
Val set: Average loss: 1.173, MNist Accuracy:97.5, Sum_Accuracy:21.54

Epoch 4 : 
Train set: Average loss: 1.1098
Val set: Average loss: 1.125, MNist Accuracy:97.94, Sum_Accuracy:29.7

Epoch 5 : 
Train set: Average loss: 1.0655
Val set: Average loss: 1.071, MNist Accuracy:98.22, Sum_Accuracy:40.3

Epoch 6 : 
Train set: Average loss: 1.0745
Val set: Average loss: 1.013, MNist Accuracy:98.46, Sum_Accuracy:49.68

Epoch 7 : 
Train set: Average loss: 0.9748
Val set: Average loss: 0.944, MNist Accuracy:98.72, Sum_Accuracy:53.14

Epoch 8 : 
Train set: Average loss: 0.8799
Val set: Average loss: 0.866, MNist Accuracy:98.62, Sum_Accuracy:64.08

Epoch 9 : 
Train set: Average loss: 0.8067
Val set: Average loss: 0.792, MNist Accuracy:98.74, Sum_Accuracy:71.84

Epoch 10 : 
Train set: Average loss: 0.7077
Val set: Average loss: 0.705, MNist Accuracy:98.82, Sum_Accuracy:79.88

Epoch 11 : 
Train set: Average loss: 0.6733
Val set: Average loss: 0.628, MNist Accuracy:98.78, Sum_Accuracy:83.54

Epoch 12 : 
Train set: Average loss: 0.5928
Val set: Average loss: 0.547, MNist Accuracy:98.84, Sum_Accuracy:88.86

Epoch 13 : 
Train set: Average loss: 0.5122
Val set: Average loss: 0.475, MNist Accuracy:98.86, Sum_Accuracy:92.52

Epoch 14 : 
Train set: Average loss: 0.4575
Val set: Average loss: 0.404, MNist Accuracy:98.8, Sum_Accuracy:95.76

Epoch 15 : 
Train set: Average loss: 0.3691
Val set: Average loss: 0.344, MNist Accuracy:98.92, Sum_Accuracy:96.88

Epoch 16 : 
Train set: Average loss: 0.2877
Val set: Average loss: 0.296, MNist Accuracy:98.92, Sum_Accuracy:97.14

Epoch 17 : 
Train set: Average loss: 0.2791
Val set: Average loss: 0.251, MNist Accuracy:99.02, Sum_Accuracy:97.56

Epoch 18 : 
Train set: Average loss: 0.2175
Val set: Average loss: 0.216, MNist Accuracy:99.08, Sum_Accuracy:97.84

Epoch 19 : 
Train set: Average loss: 0.1887
Val set: Average loss: 0.187, MNist Accuracy:99.0, Sum_Accuracy:98.26

Epoch 20 : 
Train set: Average loss: 0.1829
Val set: Average loss: 0.165, MNist Accuracy:98.92, Sum_Accuracy:98.26

Epoch 21 : 
Train set: Average loss: 0.1572
Val set: Average loss: 0.148, MNist Accuracy:99.08, Sum_Accuracy:98.3

Epoch 22 : 
Train set: Average loss: 0.1076
Val set: Average loss: 0.131, MNist Accuracy:99.02, Sum_Accuracy:98.36

Epoch 23 : 
Train set: Average loss: 0.1987
Val set: Average loss: 0.119, MNist Accuracy:99.2, Sum_Accuracy:98.6

Epoch 24 : 
Train set: Average loss: 0.1278
Val set: Average loss: 0.108, MNist Accuracy:99.34, Sum_Accuracy:98.68

Epoch 25 : 
Train set: Average loss: 0.1189
Val set: Average loss: 0.099, MNist Accuracy:99.2, Sum_Accuracy:98.78
