Explanation for Session 5 Assignment
Create a GN/BN/LN Model and its accuracies in single model.py and plot a graph together.
5.	write an explanatory README file that explains:
1.	what is your code all about,
    
     Created a single model.py file which includes GN/LN/BN and taken as an argument to decide which normalization to include in the model. Normalization type        is passed as a argument in the model.  
    
2.	how to perform the 3 normalizations techniques that we covered.
    
    3 different normalization techniques are passed as an argument to the model. By using self.normalize, to helps the model to perform better.

3.	your findings for normalization techniques,
    
    In Batch Normalization, we compute the mean and standard deviation across the various channels for the entire mini batch. In Layer Normalization, we compute     the mean and standard deviation across the various channels for a single example.
    
    GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes. Essentially, GN takes away the dependance on         batch size for normalization and in doing so mitigates the problem suffered by BN.
    
4.	add all your graphs:

![image](https://user-images.githubusercontent.com/8513086/215546500-137954ba-1182-4673-a015-35f15fd2b571.png)


5.	your 3 collection-of-misclassified-images: 

![image](https://user-images.githubusercontent.com/8513086/215552687-f4e5e28b-0c0f-4c05-bae5-1db134554ccb.png)
![image](https://user-images.githubusercontent.com/8513086/215552760-005399ae-945f-412d-afec-5fc2b1259f32.png)
![image](https://user-images.githubusercontent.com/8513086/215552895-db474957-48f4-4898-ab5d-fbed2142abff.png)
![image](https://user-images.githubusercontent.com/8513086/215552946-21be70a6-af31-43aa-8289-e39f764513a2.png)


![image](https://user-images.githubusercontent.com/8513086/215546354-37dddf89-5e11-46e9-94f4-cef60da813d2.png)



