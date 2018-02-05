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