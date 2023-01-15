mention the data representation


must mention your data generation strategy (basically the class/method you are using for random number generation)
def __getitem__(self, index):
    image = self.MNISTDataset[index][0]
    label = self.MNISTDataset[index][1]
    randomNo = random.randint(0,9)

    #Create one hot encoding for random number 
    one_hotrandomNo = torch.nn.functional.one_hot(torch.arange(0, 10))

    #add actual label and random number
    sum = label + randomNo
    return image, label, one_hotrandomNo[randomNo], sum


must mention how you have combined the two inputs (basically which layer you are combining)
 #concatenate second input to the output from above convolution
        x1 = torch.cat((x, randomNumber), dim=1)
        
must mention how you are evaluating your results 

must mention "what" results you finally got and how did you evaluate your results
        # compute the loss occured
        mnist_loss = F.nll_loss(output, target)
        addition_loss = F.nll_loss(sum_output, sum)
        loss= (mnist_loss + addition_loss)/2

        epoch_loss += loss.item()
        
        
must mention what loss function you picked and why!

Negative Loglihood Loss Function. 

