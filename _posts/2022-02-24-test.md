# This is chapter 4 of Deep Learning for Coders with fastai and PyTorch

Here's the table of contents:

1. TOC
{:toc}

## Computer vision begins with Pixels
The first step in understanding what is going on inside a computer vision model is to comprehend how computer interact with images. This chapter tries to explain the foundation of computer vision conducting some experimentations on the very popular data set MNIST, which contains images of handwritten digits collected by the National Institute of Standards and Technology by *Yann Lecun* and his Colleagues. Therefore, we need to download the sample dataset, but before that we should know how install the `fastbook` contents, and how to import `fastai` library.

```python
# installing fastbook contents and import it
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```

     |████████████████████████████████| 720 kB 8.9 MB/s 
     |████████████████████████████████| 189 kB 24.3 MB/s 
     |████████████████████████████████| 48 kB 5.8 MB/s 
     |████████████████████████████████| 1.2 MB 36.9 MB/s 
     |████████████████████████████████| 55 kB 4.0 MB/s 
     |████████████████████████████████| 51 kB 380 kB/s 
     |████████████████████████████████| 558 kB 55.2 MB/s 
     |████████████████████████████████| 130 kB 56.7 MB/s 
     Mounted at /content/gdrive

```python
# import libraries: fastbook, fastai, and pandas
from fastbook import *
from fastai.vision.all import *
import pandas as pd
```

In a computer, everything is represented as a number, therefore, to view the numbers that make up this image, we have to convert it to a `NumPy` array or `PyTorch` tensor. The following contents introduce some tricks in working with these two data structures, but if you want to know more about them you can refer to the following links:

- [NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
- [PyTorch tensors](https://pytorch.org/docs/stable/tensors.html)

## Basic Image Classifier Model
This section of the book comes up the idea of using the pixel similarity as the very basic method to classify images. For this purpose, the average pixel value for every pixel of the 3s samples, and the for 7s samples are calculated. As a result, we have two arrays/tensors containing the pixel values for two images that we might call the "ideal" 3 and 7. Hence, to classify an image as 3 or 7, we can evaluate which of these two ideal digits the image is similar to.

### Constructing the base model
**Step 1**: Calculating the average of pixel values of each of two sample groups of 3s and 7s. Creating a tensor containing all of our 3s stacked together. For this, Python list comprehension is used to create a plain list of the single image tensors.
```python
# creating a tensor containing all of 3s sample images stacked together 
# using list comprehension
three_tensors= [tensor(Image.open(img)) for img in threes]

# and the same for 7s sample images
seven_tensors = [tensor(Image.open(img)) for img in sevens]

# checking the number of items in each tensor
len(three_tensors), len(seven_tensors)
```
     (6131, 6265)

**Step 2**: Stacking up all the image tensors in this list into a single three-dimensional tensor (rank-3 tensor) using PyTorch stack function. The values stored in stacked tensor is casted to float data types, as required by some PyTorch operations, such as taking a mean.
```python
# stacking up all the image tensors in the list in to one rand-3 tensor, and 
# cast it to float types.
stacked_threes = torch.stack(three_tensors).float() / 255
stacked_sevens = torch.stack(seven_tensors).float() / 255

# checking the stacked tensor's size
stacked_threes.shape
```
     torch.Size([6131, 28, 28])

{% include info.html text="<strong>Definition</strong>:
<br>Tensor's rank is the number of axes or dimensions in a tensor, shape is the size of each axis of a tensor." %}

```python
# rank or dimension of a tensor using len()
len(stacked_threes.shape)

# or directly using ndim
stacked_threes.ndim
```
     3

```python
stacked_threes.shape[1]
```
     28

**Step 3**: Computing what the ideal 3 and 7 look like through calculating the mean of all the image tensors by calling mean for every index position over images along dimension 0 of both stacked rank-3 tensor.
```python
# calculating mean for index position over the images along dimension 0 
# of the both rank-0 stacked tensors of 3s and 7s
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)

#showing ideal 3
show_image(mean3)
```

![ideal 3](/mytechblog/images/2022-02-24-DL-fastbook-chapter04/ideal_3.png "ideal 3")

### Defining Loss Function
Now that the 3-or-7 classifier model is ready to use, we can pick up an arbitrary 3 and calculate its distance from the "ideal digits".
First we need to select an arbitrary image, e.g. from 3s sample collection, as shown below:

```python
# select an arbitrary 3 from 3s stacked tensor
a_3 = stacked_threes[1]

# showing the selected image
show_image(a_3)
```
![selected 3](/mytechblog/images/2022-02-24-DL-fastbook-chapter04/arbitrary_3.png "selected 3")

Two main alternatives to measure distance in this case are as:
* **MAE**: the Mean absolute value of difference, which is also called *L1 norm*.
```python
# calcuting the distance from ideal 3 using MEA and RSME
dist_3_mea = (a_3 - mean3).abs().mean()
dist_7_mea = (a_3 - mean7).abs().mean()
dist_3_mea, dist_7_mea
```
     (tensor(0.1114),  tensor(0.1586))
    

* **RSME**: root mean squared root, which is also called *L2 norm*.
```python
from hashlib import sha3_384
# calculating the distance from ideal 7 using MEA and RSME
dist_7_rsme = ((a_3 -mean7) ** 2).mean().sqrt()
dist_3_rsme = ((a_3 - mean3) ** 2).mean().sqrt()
dist_3_rsme, dist_7_rsme 
```
     (tensor(0.2021), tensor(0.3021))

* PyTorch provides both of measures mentioned above as **loss functions**, which can be found inside  `torch.nn.functional` (recommended by PyTorch team to be imported as `F`). This function is also available in `fastai` by default as `F`.

```python
# calculating the distance from ideal 3 using loss functions in fastai
F.l1_loss(a_3.float(), mean3), F.mse_loss(a_3, mean3).sqrt()
```
     (tensor(0.1114), tensor(0.2021))

```python
# calculating the distance from ideal 7 using loss functions in fastai
F.l1_loss(a_3.float(), mean7), F.mse_loss(a_3, mean7).sqrt()
```
     (tensor(0.1586), tensor(0.3021))

<strong>Conclusion</strong>: The values calculated for the <strong>L1 norm (MEA)</strong> and the <strong>L2 norm (RSME)</strong> show that the distance between the selected 3 and the "ideal 3" is less than the distance to the ideal 7. Hence, we can conclude that even this simple classifier model is working well in providing us with the correct prediction in this case.


### Metrics Computation Using Broadcasting
*Metric* is a numeric measure that enables us to evaluate our model's performance. It is calculated based on the number of labels in our dataset that are predicted by the model.
Taking the average of values calculated using each of the loss functions explained in the previous section, i.e. MEA and RSME, over the whole dataset can be used as a metric for a model. However, using the model's *accuracy* as the metric for classification models is more common since neither MAE nor RSME seems very easy to understand to most people.<br>
Metrics are calculated over the validation set to prevent the model from overfitting. Model overfitting may not be a risk the 3-or-7 basic classifier model explained in the previous section because it does not adopt any training component. Conversely, overfitting poses a major concern in training every model in machine learning and deep learning.<br>
To calculate the accuracy of our simple classifier model, we need to instantiate tensors for 3s and 7s that are taken out form validation set. In MNIST dataset, the validation set is placed on a seperate directory named `valid`. So, the scripts for creating the validation tensors will be as shown below:


```python
# validation tensor for 3s samples from validation dataset
valid_3_tens = torch.stack([tensor(Image.open(img)) 
                          for img in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float() / 255

# validation tensor for 7s samples from validation dataset
valid_7_tens = torch.stack([tensor(Image.open(img)) 
                          for img in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens / 255

#checking the validation tensors' shape
valid_3_tens.shape, valid_7_tens.shape
```
  (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))



To calculate a metric for overall model accuracy in detecting 3, we will firstly need to define a function that calculate the distance for every image in the validation set. When applying this function to validation tensor, the distance from ideal 3 will be calculated for every single image using tensor's *broadcasting* feature as shown below:


```python
def mnist_dist(a, b):
  return (a-b).abs().mean((-1, -2))

valid_3_dist = mnist_dist(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
```
  (tensor([0.1263, 0.1413, 0.1430,  ..., 0.1332, 0.1471, 0.1469]),
 torch.Size([1010]))



In broadcasting, PyTorch automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank. There are two worth-mentioning point about broadcasting as below:
*   Expanding the lower-randed tensor does not mean that PyTorch copies mean3 1010 times or allocate any additional memory to do this, it actually pretends to have a tensor of that shpape.
*   PyTorch performs the calculation in C (or CUDA if we use GPU), and this the secret of thousands of times lower computation time.
<br>Now we need a function to verify whether or not each of the images is 3 by comparing its distrance from ideal 3 and ideal 7.


```python
def is_3(x):
  return mnist_dist(x, mean3) < mnist_dist(x, mean7)

# correctness percentage of predicting 3 
accuracy_3s = is_3(valid_3_tens).float().mean()

# correctness percentage of detecting 7 (non 3 image).
accuracy_7s = (1 - is_3(valid_7_tens).float().mean())

accuracy_3s, accuracy_7s, (accuracy_3s + accuracy_7s) / 2
```
  (tensor(0.9168), tensor(0.9854), tensor(0.9511))

The result above shows an accuracy over 90% on both predicting correctly 3s and correctly detecting 7s as not beeing a 3, which is quite accetable for such simple classifier model.


## Stochastic Gradient Descent (DSG)


> <strong>Definition of machine learning according to Arthur Samuel</strong><br> Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a  mechanism for altering the weight assignment so as to maximize the performance.  We need not go into the details of such a procedure to see that it could be made entirely automatic and see that a machine so programmed would “learn” from its experience.

To enable our simple 3-or-7 classifier to take advantage of the power of deep learning, we will first have to represent it based on Samuel's definition, Instead of trying to find similarity between an image and an "ideal image", a set of weights can be set for each individual pixel, such that the highest weights are assiciated with those pixels most likely to be black for particular category. This can be represented as a function and set of weight values for each possible category, e.g., the probability of being the number of 8:

```python
def prob_eight(x, w) = (x * w).sum()
```

Assuming that X is the image, represented as a vector, i.e. all the rows stacked up end to end into a single lone line, and W is the vector of the weights. We aim to search for a specific set of values for W that result in maximizing the function value for those images that are 8s, and in minimizing it for those images that are not.

>The steps required to turn this function into amachine learning classifier:
1.   *Iinitialize* the weights
2.   For each image, use these weights to *predict* whether it is likely to be a 3 or a 7.
3. Based on these predictions, calculate how good the model is (its *loss*)
4. Calculating the *gradient*, which measures the impact of change in each weight on the its pertinent loss.
5. *Step*, making decision between change all the weights based on the calculation, or leave it unchanged and terminate the search process.
6. Go back to step 2 and *repeat* the process.
7. Iterate until the stop criteria are met (e.g. the model is well-trained enough or the waiting time has been exceeded) are met.

<strong>Initialize</strong>: setting the parameters to random values.<br>
<strong>Loss</strong>: testing the effetiveness of any current weight assignment in terms of actual performance.<br>
<strong>Step</strong>: to figure out whether a weight should be increased/descressed a bit. Basically, this may most likely to be too slow, but calculating *gradient* assist us to comprehend in which direction, and by approximately how much, to change each weight.<br>
<strong>Stop</strong>: deciding how many epochs to train the model for.

This section of the book tries to explain the concept of how calculating greadient can help us optimize the learning proces of a deep learning model with defining a quadratic function as loss function as shown below:

```python
def f(x):
  return x ** 2

plot_function(f, 'x','x**2')
```
![quadratic function plot](/mytechblog/images/2022-02-24-DL-fastbook-chapter04/quad_func_plot_01.png)

Based on the *initialize* step defined above, if we select a random value for a parameter, calculating the value of loss results in having a point on the plot as seen below:

```python
plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(1.5), color='red')
```
![quadratic function plot 1](/mytechblog/images/2022-02-24-DL-fastbook-chapter04/quad_func_plot_02.png)

The following plot shows that, if we decide to make a small *adjustment* (step) to the parameter, the slope indicates the displacement of the point.<br>

![quadratic function plot 3](/mytechblog/images/2022-02-24-DL-fastbook-chapter04/quad_func_plot_03.png)

In view of what has been demonstrated above, we can conclude that it is possible to reach the lowest point on our curve by adjusting our weight a little in the direction of the slope, calculating the loss, and making a few adjustments. This basic idea dates back to Isaac Newton, who noted that we can optimize arbitrary functions in this way regardless of their complexity.

![quadratic function plot 4](/mytechblog/images/2022-02-24-DL-fastbook-chapter04/quad_func_plot_04.png)

>The gradients indicates the slope of our function, and do not specify exactly how far to adjust parameters. They can, however, give us an indication of how far, so as if the slope is very large, that may suggest more adjustments to do, whereas if the slope is very small, that may suggest that we are close to the optimal value.

### Calculating Gradient
Deep learning models are opitmized by calculating the *gradient* which indicates how much we need to adjust each weight to improve the model.
In mathematics, a function's *gradient* is simply another function, whereas in deep learning, *gradient* usually refers to the value of the function's derivative at a given argument value.

>Some people may concerns about complication of calculating gradient, but a good news is that PyThorch take the budden by automatically computing the derivative of nearly any function!


Let's see how PyTorch makes life more easier for us by calculating greadient. We start with pick a tensor value and let PyTorch know that we want to calculate gradient with respect to the variable x at that value by calling the function `requires_grad_`. This way, when you ask PyTorch to calculate the gradient, it will remember to care about other direct calculations.

```python
xt = tensor(3.).requires_grad_()

# calculating the function value with x value of 3.0
yt = f(xt)
yt
```
     tensor(9., grad_fn=<PowBackward0>)

Not only does the result show the function value, but it also shows that it has a gradient function that can be called to calculate the gradient when needed.

Now, we ask PYTorch to calculate the gradient and show it by retrieving `grad` property of `x` as shown below:
```python
yt.backward()

xt.grad
```
     tensor(6.)

As we see, it return the value of the derivative of the quadratic function x ** 2, which is 2 * x.

{% include alert.html text="<strong>Don't forget</strong>:
<br>When calling grad, PyTorch exhausts the gradient function, and reusing it will cause an error. So, in prior to reusing grad the the function f(xt) must be re-calculated, i.e., yt = f(xt)." %}<br>

>The *"backward"* here refers to backpropagation, which the name given to the process of calculating the derivative of each layer. This is called backward pass of the network, as opposed to forward pass, which is where the activation are calculated.

When we want to calculate the gradient for a vector, we have add sum to our function to return a scalar as a value to function caller.
```python
def f(x):
  return (x ** 2).sum()

xt = tensor(3., 4., 10.).requires_grad_()
yt = f(xt)
yt
```
     tensor(125., grad_fn=<SumBackward0>)

And our gradient will be as shown below:
```python
yt.backward()
xt.grad
```
     tensor([ 6.,  8., 20.])

### Stepping with a Learning Rate
Most approaches to deep learning come up with basic idea of mutiplying the gradient by some small number, called the *learning rate* (LR). Although the learning rate can be set to any arbitrary value, it is commonly set to number between 0.001 and 0.1.
