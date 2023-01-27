Code1 - Setup


Target

	. Set up the skeleton with dropout, batch Norm etc

Results

	. Parameters - 9368

	. Best training accuracy- 98.57

	. Best test accuracy - 99.11

Analysis

	. Model is under-fitting

	. The gap between test and train is high

	. Capacity can be increased

Code2

Target

	. Changed the in and output channels, batch norm 

	. Changed the number of kernels in some conv layers

	. Dropout - 0.15

Results

	. Parameters - 6656

	. Best training accuracy- 98.44

	. Best test accuracy - 99.02

Analysis

	. Target is not reached, 

	. Model is still under-fitting


Code 3

Target

	. Increased the dropout from 0.05

Results

	. Parameters - 6656

	. Best training accuracy- 98.88

	. Best test accuracy - 99.23

Analysis

	. Model is still under-fitting

	. The accuracy of both is reduced by increasing dropout


Code 4

Target

	. Dropout from 0.001

Results

	. Parameters - 9680

	. Best training accuracy- 99.30

	. Best test accuracy - 99.40

Analysis

	. Reached the desired accuracy at 12th epoch

	. Should see how it works introducing LR and reducing no of parameters


Code 5

Target

	. Dropout - 0.01

Results

	. Parameters - 6766

	. Best training accuracy- 99.27

	. Best test accuracy - 99.38

Analysis

	. Reached the desired accuracy at 14th epoch

