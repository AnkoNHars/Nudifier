<h1>CycleGan</h1>

<h3>Model Presentation</h3>

CycleGan is a style transfer model, it makes image to image translation trainnig on unpaired data.
Paper: https://arxiv.org/abs/1703.10593
Used Implementation: https://github.com/junyanz/CycleGAN/blob/master

<h3>Results</h3>:

####Phase 1: 
Test with full datatset of 5000 pictures of nude girls extracted from pornhub and 5000 pictures of dressed girls extracted from instagram
Trainning 200 epochs:

<img src="https://github.com/AnkoNHars/Nudifier/blob/master/CycleGan/Full%20images/results/0.jpg">
<img src="https://github.com/AnkoNHars/Nudifier/blob/master/CycleGan/Full%20images/results/4.png">

####Conclusion:
The dataset was too small and to diverse to allow the model to extract the real differences between dressed and nude people

####Phase 2:
Test with the extraction of the persons on the images using Mask-RCNN

<img src="https://github.com/AnkoNHars/Nudifier/blob/master/CycleGan/Cropped%20Images/results/3.jpg">

####Conclusion:
Diffences of poses of people between the distributions biased the dataset
