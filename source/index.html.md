---
title: API Reference

language_tabs: # must be one of https://git.io/vQNgJ
  - python

toc_footers:
  - <a href='https://github.com/iporollo/CSE5914-Tech-Tutorial'>See the Code on Github</a>
  - <a href='https://www.tensorflow.org/'>TensorFlow Documentation</a>


includes:
  - real-time-object-detection

search: true
---

# Introduction

Welcome to the TensorFlow tutorial created by Sprint4. 

In this tutorial, we plan to give you a high level overview of what TensorFlow actually is. 

Apart from describing TenserFlow, we also want to show you how to use it in real world projects. We provided two real world examples (with code) that can be found after the overview. 

We hope you enjoy this gentle introduction to TensorFlow.

# What is TensorFlow?

> To install Tensorflow on your machine use:

```python
$ pip install tensorflow
```

TensorFlow is a python based library which provides various machine learning and deep learning (neural network) models & algorithms. 

It is mainly used for numerical computation and large scale machine learning. 

# What Are Tensors?

The term ‘Tensor’ basically represents data in deep learning. Tensors are simply multi-dimensional arrays which allow you to represent data that has higher dimensionality. 

In large scale data sets, the dimensions of the tensors (data) are often referred to as having different features. 

The term ‘TensorFlow’ or ‘the flow of tensors’ comes from the operations which deep learning models (neural networks) perform on tensors.

# Basics

## Basic Program

> In your console, run python to launch the interpreter like so:

``` python
$ python
```

> Example 1 (don't include the '>'):

``` python
> import tensorflow as tf
> tf.enable_eager_execution()
> tf.add(1, 2).numpy()
3
```

> Example 2 (don't include the '>'):

``` python
> hello = tf.constant('Hello, TensorFlow!')
> hello.numpy()
'Hello, TensorFlow!'
```

We start off with a gentle introduction to the TensorFlow language by using it in the python interpreter. 

Launch the interpreter by running the `python` command. 

In the intpreter, add the three lines of code in the first example.

After adding the three lines, the python interpreter will return 3. TensorFlow uses the `.add()` function to add the two parameters we pass in. 

Afterwards, we need to convert it to readable format with `.numpy()`

Next, enter the two lines of code in example 2. 

As you can see, the interpreter returns the assigned constant value as `'Hello, TensorFlow'`

## Computational Graphs

> Create a python file and insert:

```python
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
``` 

TensorFlow uses computational graphs for its operations. A computational graph is made up of various TenserFlow operations arranged as nodes on a graph. 

Each node takes 0 or more tensors are input and results in a tensor as the output. 

As an example, observe the code given. 

`a` is a TensorFlow constant with the value of `5.0`. 

`b` is a TensorFlow constant with the value of `6.0`. 

The two values are both tensors in the computational graph. 

`c` connects the two tensors together by a multiplication operation. The product of `a` and `b` is now a new node in the graph `c` with the resulting the tensor stored in it. 

Essentially, a computational graph is just an alternate way of conceptualizing the mathematical calculations that take place in a TensorFlow program. 

The operations assigned to different nodes of a Computational Graph can be performed in parallel, thus, providing a better performance in terms of computations.

### Running the Graph

```python
sess = tf.Session()
output_c = sess.run(c)
print(output_c)
sess.close()
```

> Output: 30

To see the resulting value of `c`, we need to run the created computational graph within a TenserFlow session. 

Sessions place the graph onto devices (CPUs or GPUs) and provide methods to execute the graph.

When using a session, it encapsulates the control and state of the TensorFlow runtime. 

Basically, this means that it stores the information about the order in which all the operations will be performed and passes the result of an already computed operation to the next operation in the pipeline.

Observe the provided example. In the example we create a new TensorFlow session and use it to compute the orginal `a`, `b`, `c` computational graph (provided in the previous example).

In the example, the `tf.Session()` function creates a new TensorFlow session object. 

`sess.run(c)` runs the computational graph within a session and stores the output to the variable `output_c`.

Afterwards, we print the result of the variable and close the session with `sess.close()`

# Technicalities

In TensorFlow, constants, placeholders and variables are used to represent different parameters of a deep learning model.

## Constants

Constant nodes have no parameters meaning they take zero input. They only store constant values.

In the previous section’s example, `a` and `b` were constant nodes with the values of `5.0` and `6.0` respectively. 

Constants are initialized when you call `tf.constant()`, and their value can never change.

## Placeholders

```python
import tensorflow as tf
 
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

product = a*b
 
sess = tf.Session()
 
result = sess.run(product, {a: [1,3], b: [2, 4]})
print('Multiplying a b:', result)
```

> Output: Multiplying a b: [2 12]

A TensorFlow placeholder allows computational graph nodes to take external inputs as parameters. 

Essentially, a placeholder makes a promise to provide a value to the computational graph either later or during runtime. 

There are a few important things to remember when using placeholders. First off, they are not initialized, nor do they contain any data. 

With placeholders, the user provides inputs to the placeholder that are considered during run time. Also, if a placeholder is executed without an input, it generates an error. 

An example usage of placeholders is provided. 

As you can see, the `tf.placeholder()` function creates the TensorFlow placeholder. `tf.float32` is one of TensorFlows’s own data types. 

After creating the placeholders, we assign the multiplication operation of `a` and `b` to the node `product`. 

Afterwards, we create a new `Session` object and execute the `product` node. Inside the `sess.run()` function, we pass `[1,3]` to the placeholder `a` and `[2,4]` to the placeholder `b`. 

As stated previously, the data is passed to the placeholders during run time. 

The resulting output is `[2, 12]`

## Variables

```python
var = tf.Variable( [0.4], dtype = tf.float32 )

init = tf.global_variables_initializer()
sess.run(init)
```

Variables give TensorFlow the ability to modify the computational graph such that it can produce new outputs with respect to the same inputs. While it may seem a bit confusing, all variables really do is allow you to add parameters (nodes) to the computational graph that are trainable (can be changed over time).

In the example, you can observe how to define a variable using the `tf.Variable()` function.

Variables aren’t truly initialized when you declare them. To initialize a variable, you need to explicitly call the functions given in the code below the variable declaration. 

**It is necessary to initialize variables before the computation graph is used.**

<aside class="notice">
This is the end of the general TensorFlow overview. The following two sections contain real world applications of TensorFlow.
</aside>

# Linear Regression Example

## Initial Setup

> Create a new file for this example and add the following code:

```python
w = tf.Variable([.4], tf.float32)
b = tf.Variable([-0.4], tf.float32)
x = tf.placeholder(tf.float32)
 
linear_regression_model = w * x + b
 
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
 
print(sess.run(linear_regression_model {x: [1, 2, 3, 4]})) 
```

> Output: [ 0 0.40000001 0.80000007 1.20000005]

Linear regression is an approach to modeling the relationship between dependent and independent variables. Relationships are modeled using linear predictor functions, whose unknown model parameters are estimated from the data.

The linear regression equation is as follows:

`y = b + wx`

Where:

- `y` is the dependent variable
- `b` is the bias (y-intercept)
- `w` is the slope variable
- `x` is the independent variable

In the code give, you can observe the various value assignments to the variables. 

- `y` is the resulting linear regression model derived from the operations on the other variables
- `b` is declared as a variable with the initial value of `-0.4`
- `w` is declared as a variable with the initial value of `0.4`
- `x` is declared as a placeholder for providing independent variables during runtime

Afterwards, we create a session and intialize the variables. Then, we run the session with our independent variables passed in. 

## Loss Function

```python
y = tf.placeholder(tf.float32)
error = linear_regression_model - y
squared_errors = tf.square(error)
loss = tf.reduce_sum(squared_errors)
print(sess.run(loss, {x:[1,2,3,4], y:[2, 4, 6, 8]})
```

> Output 90.24

We will use a loss function to measure how far apart the current output of our model is from that of the desired or target output. 

We will use a Mean Squared Error (MSE) loss function for our model. The loss function will be calculated with respect to the model output `linear_model` and the desired target output.

We will use the following equation for the loss function:

`E = 0.5 * (target - prediction) ^ 2`

- `E` is the mean squared error result
- `target` is the target output
- `prediction` is the actual model prediction

Observe the code given.

First off, we set a placeholder for the desired output. 

Next, we get the squared error by using the `t.square()` function. 

Afterwards, we can get the mean squared error by using the `tf.reduce_sum()` function. 

Finally, we run the session, passing in our independent and expected values.

The output of this is a high loss value. In order to lower this loss value, we need to adjust the weight and bias of the linear regression model.

## Training the Model

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
 
train = optimizer.minimize(loss)
 
for i in range(1000):
     sess.run(train, {x:[1, 2, 3, 4], y:[2, 4, 6, 8]})
print(sess.run([W, b]))
```

> Output: [array([ 1.99999964], dtype=float32), array([ 9.86305167e-07], dtype=float32)]

TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function (error). The simplest optimizer is gradient descent. 

Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. The steepest descent is determined according to the magnitude of the derivative of the loss. 

Observe the given code where we add gradient descent to our model.

First, we create an instance of the gradient descent optimizer using the `tf.train.GradientDescentOptimizer()` function. We pass in a parameter of `0.1` which is the learning rate of the optimizer.

Afterwards, we minimize the loss of our model using the optimizer with the `optimizer.minimize()` function. This function will minimize the loss by modifying the model parameters `w` and `b`.
 
Then, we run the `train` node 1000 times to change the `w` and `b` variables to optimized values. 

Finally, we print out the optimal `w` and `b` to get the best performance on our linear regression model. 

We now have successfully created and trained a linear regression model using TensorFlow. Similar steps can be followed to create various other models for data that you work with. 


