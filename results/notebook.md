# 18/05/18: Second try for linear models (Marine)

The code contained a bug in the `def inference(images)` function in  [`cifar10.py`](../src/cifar10.py). The tensorflow function `tf.nn.top_k` does a *argsort*, i.e, orders the indices of the elements of a vector so that they are sorted (in increasing or deacreasing order). What we want instead is the rank of each element of the vector. We corrected the bug by chaining two `tf.nn.top_k` which produces what we want, although it is probably not the most efficient way to do it.

The previous version of the code was equivalent to exchanging the roles of `f` and `w`. As a result, `w` was initialised to an increasing target quantile instead of `f`. That explains the poor results obtained previously with QN. The new results can be summarized as follows, picking the best parameters for each configuration:  

|| greyscale | RGB|
|:---:|:---:|:---:|
|Original data|0.306|0.367|
|QN (uniform)|0.247|0.367|
|SUQUAN (uniform initialisation)|0.248|0.372|

We observe that on CIFAR10, quantile normalization to uniform distribution decreases the accuracy by 6% for greyscale images but does not affect the performance for rgb images. In addition, SUQUAN is able to slightly improve the performance of QN on RGB (+0.5%), but not on greyscale. Similar results were obtaiend by using the *normal* target quantile instead of the *uniform* one. 

To see wether or not the quantile function `f` actually changes during training, we ran additional experiments where the initial target quantile is set to a constant vector, and monitored `f` with tensorboard during training, as well as the test accuracy. The script [`runexamples`](180518/runexamples) contains the parameters used for these additional experiments. In both experiments, we observed that the test accuracy starts at 10% (the accuracy of a random classifier) and then slowly increases to reach 0.362 (RGB) and 0.234 (greyscale). The learned target quantile are increasing vectors. They are quite smooth in a first part of the training and then become more and more noisy.

These experiments show that SUQUAN does learn something, but it seems that it can't do better than the original data in these cases.


# 18/02/03: First try (jp)
We started from the very nice [Tensorflow tutorial on CIFAR-10 image classification with a convolutional neural network](https://www.tensorflow.org/tutorials/deep_cnn) a modified the code to implement various versions of SUQUAN:

* a linear model on images after quantile normalization of their pixel intensities, as described in the [original SUQUAN paper](https://arxiv.org/abs/1706.00244), with the possibility to optimize the quantile function together with the linear model.
* an extension to neural network (hence *deepSUQUAN*) where the quantile normalization is performed on the images at the first layer. Note that conceptually, this could be done on any layer, as a particular normalization.

The implementation of SUQUAN is essentially in the definition of the model, in the `def inference(images)` function in  [`cifar10.py`](../src/cifar10.py).

We added a few options to the command line:

* `--use_linear_model` (True or False) to use a linear model or the CNN
* `--use_grey_scale` (True or False) to transform images into greyscale or not
* `--use_suquan` (True or False) to perform quantile normalization
* `--optimize_f` (True or False) to optimize the quantile function or not, if `--use_suquan == True`
* `--Wwd` (float) the weight decay (a.k.a. L2 penalty) of the linear model if `--use_linear_model == True`
* `--Fwd` (float) the weight decay (a.k.a. L2 penalty) of the quantile function if `--use_suquan == True` and `--optimize_f == True`

The script [`runall`](180203/runall) runs a series of experiments by varying the different parameters, for the linear model. Results are stored in [`results.txt`](180203/results.txt), showing the @1 accuracy on the test set. We can summarize them as follows, picking the best parameters for each configuration:

* Original data: 0.301 (greyscale), 0.372 (RGB)
* QN (uniform): 0.208 (greyscale), 0.205 (RGB)
* SUQUAN (with uniform initialization): 0.208 (greyscale), 0.214 (RGB)

We observe that on CIFAR10, quantile normalization to uniform distribution decreases the accuracy by 10 (greyscale) or even 17% (RGB). In addition, SUQUAN is able to improve a bit the performance of QN on RGB (+1%), but not on greyscale.

Next steps:

* test deepSUQUAN
* check why QN is so bad (what information is there in the intensity distribution?)