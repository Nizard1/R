library(keras)
library(sigmoid)

mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Preparing the image data 

# Before training, we'll preprocess the data by reshaping it
# into the shape the network expects and scaling it so that 
# all values are in the [0, 1] interval. Previously, our 
# training images, for instance, were stored in an array
# of shape (60000, 28, 28) of type integer with values in 
# the [0, 255] interval. We transform it into a double 
# array of shape (60000, 28 * 28) with values between 0 and 1. 

train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

#  We also need to categorically encode the labels, a step that's explained in 
#  Preparing the labels 
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

dim(train_images)
dim(test_images)
length(dim(train_images)) # we display the number of axes of the tensor train_images: 2D
typeof(train_images)



# The network architecture 
network <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")

# pipe operator (%>%) used to invoke, , and so on.  
# methods on the network object
# For now, read it in your head as "then":
# start with a model, then add a layer,
# then add another layer, and so on. 

#  The compilation step 
network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Finally, this was the training loop:
network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

# We quickly reach an accuracy of 0.989 (98.9%) on the training data.
# Now let's check that the model performs well on the test set, too: 

metrics <- network %>% 
  evaluate(test_images, test_labels)

metrics
# Let's generate predictions for the first 10 samples of the test set:
network %>% predict_classes(test_images[1:10,])
###################################################

# To make this more concrete, let's look back at the data we processed in the MNIST example. 
# First, we load the MNIST dataset:
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

#str(train_images)
#str(train_labels)
dim(train_images)
length(dim(train_images)) # we display the number of axes of the tensor train_images:3D
typeof(train_images)

# So what we have here is a 3D tensor of integers. More precisely,
# it's an array of 60,000 matrices of 28 × 28 integers. Each such
# matrix is a grayscale image, with coefficients between 0 and 255
# The fifth sample in our dataset 
digit <- train_images[7,,]
plot(as.raster(digit, max = 255))

my_slice <- train_images[10:99,,]
dim(my_slice)
length(dim(train_images))

my_slice <- train_images[10:99,1:28,1:28]
dim(my_slice)
my_slice <- train_images[, 15:28, 15:28]
dim(my_slice)


# The notion of data batches 
# In general, the first axis in all data tensors you'll come across
# in deep learning will be the samples axis (sometimes called the samples dimension).
# In the MNIST example, samples are images of digits

# in deep-learning models don't process an entire dataset at once;
# rather, they break the data into small batches. Concretely,
# here's one batch of our MNIST digits, with batch size of 128

batch <- train_images[1:128,,]
# here's the next batch
batch <- train_images[129:256,,]

## Real-world examples of data tensors 

# 1. Vector data-2D tensors of shape (samples, features).

# 2. Timeseries data or sequence data-3D tensors of shape (samples, timesteps, features).

# 3. Images-4D tensors of shape (samples, height, width, channels) or
#    (samples, channels, height, width)

# 4. Video-5D tensors of shape (samples, frames, height, width, channels) or
#    (samples, frames, channels, height, width)

### read more: Deep Learning with Python-François Chollet ch2.3 

# The gears of neural networks: tensor operations
# we were building our network by stacking dense layers on top of each other.
# A layer instance looks like this:
layer_dense(units = 512, activation = "relu")
# This layer can be interpreted as a function, which takes as input a
# 2D tensor and returns another 2D tensor-a new representation for the
# input tensor. Specifically, the function is as follows (where W is a
# 2D tensor and b is a vector, both attributes of the layer):

output = relu(dot(W, input) + b)
# Let's unpack this. We have three tensor operations here:
# a dot product (dot) between the input tensor and a tensor named W;
# an addition (+) between the resulting 2D tensor and a vector b; and,
# finally, a relu operation. relu(x) is max(x, 0). 


naive_relu <- function(x) {               
  for (i in nrow(x))
    for (j in ncol(x))
      x[i, j] <- max(x[i, j], 0)
    x
}
# x is a 2D tensor (R matrix).

naive_add <- function(x, y) {             
  for (i in nrow(x))
    for (j in ncol(x))
      x[i, j] = x[i, j] + y[i, j]
    x
}
# x and y are 2D tensors (matrices)
# BLAS implementation (Basic Linear Algebra Subprograms)
# if you have one installed (which you should). BLAS are low-level,
# highly parallel, efficient tensor-manipulation routines typically 
# implemented in Fortran or C. 
# sessionInfo()
# Microsoft R Open & MKL Downloads
x <- array(round(runif(1000, 0, 9)), dim = c(64,3,32,10))      
y <- array(5, dim = c(3, 2))                                    
z <- x + y         # Element-wise addition       
z <- pmax(z, 0)    # Element-wise relu    
sweep(y,2, 5, '+')
# The second argument (here, 2) specifies the dimensions of x over which to sweep (adds 5)
sweep(y,c(1,2), 5, '+') # like apply

x <- array(round(runif(1000, 0, 9)), dim = c(64, 3, 32, 10))      
y <- array(5, dim = c(32, 10))                                    
# The following example sweeps a 2D tensor over the last two dimensions
#  of a 4D tensor (3,4) using the pmax() function: 
z <- sweep(x, c(3, 4), y, pmax)   # c(3, 4) indicates rows and columns                                

# Tensor dot
# An element-wise product is done with the * operator in R,
# whereas dot products use the %*% operator
z <- x %*% y

# The dot operation, also called a tensor product
# (not to be confused with an element-wise product) is the most common, 
# most useful tensor operation

naive_vector_dot <- function(x, y) {              
  z <- 0
  for (i in 1:length(x))
    z <- z + x[[i]] * y[[i]]
  z
}
# x and y are 1D tensors (vectors)
# the dot product between two vectors is a scalar
# and that only vectors with the same number of elements
# are compatible for a dot product.
x<-c(1,3,8);
naive_vector_dot(x,c(4,5,6))
# dim(x %*% y)=0  is x %*% y is scaler 0D

# You can also take the dot product between a matrix x and a vector y,
# which returns a vector whose elements are the dot products between y 
# and the rows of x. You implement it as follows: 

naive_matrix_vector_dot <- function(x, y) {          
  z <- rep(0, nrow(x))
  for (i in 1:nrow(x))
    for (j in 1:ncol(x))
      z[[i]] <- z[[i]] + x[[i, j]] * y[[j]]
    z
}

# x is a 2D tensor (matrix). y is a 1D tensor (vector).
# Note that as soon as one of the two tensors has more than one 
# dimension, %*% is no longer symmetric, which is to say that x %*% y 
# isn't the same as y %*% x. 
x<-matrix(c(3,5,6,8,9,1),nrow=2,ncol=3);
y<-c(1,1,1); # must ncol of x = number of element of y whoch is 3
naive_matrix_vector_dot(x,y) # dot products between y and the rows of x
# or
x %*% y # dot product  in r best
# dim(x %*% y)=2  is x %*% y is Vector 1D
# Of course, a dot product generalizes to tensors with an arbitrary
# number of axes. The most common applications may be the dot product
# between two matrices. You can take the dot product of two matrices x and y
# (x %*% y) if and only if ncol(x) == nrow(y). The result is a matrix with 
# shape (nrow(x), ncol(y)), where the coefficients are the vector products
# between the rows of x and the columns of y. Here's the naive implementation
naive_matrix_dot <- function(x, y) {                   
  z <- matrix(0, nrow = nrow(x), ncol = ncol(y))
  for (i in 1:nrow(x))
    for (j in 1:ncol(y)) {
      row_x <- x[i,]
      column_y <- y[,j]
      z[i, j] <- naive_vector_dot(row_x, column_y)
    }
  z
}
# x and y are 2D tensors (matrices).
x<-matrix(c(3,5,6,8,9,1),nrow=2,ncol=3);
y<-matrix(c(3,5,6,8,9,1),nrow=3,ncol=2);
naive_matrix_dot(x,y)
# or
x %*% y #  dotproduct in r best
#   x %*% y is Vector 2D
# Note that as soon as one of the two tensors has more than one dimension,
# %*% is no longer symmetric, which is to say that x %*% y isn't the same as y %*% x. 
# x, y, and z are pictured as rectangles (literal boxes of coefficients).
# Because the rows and x and the columns of y must have the same size,
# it follows that the width of x must match the height of y.
# If you go on to develop new machine-learning algorithms,
# you'll likely be drawing such diagrams often. 

#(a, b, c, d) . (d) -> (a, b, c)  # 4D.0D=3D
#(a, b, c, d) . (d, e) -> (a, b, c, e) # 4d.2D=4D


# reshaping a tensor means rearranging its rows and columns to 
# match a target shape. Naturally, the reshaped tensor has the same 
# total number of coefficients as the initial tensor.
# Reshaping is best understood via simple examples: 
# read row-wise
x <- matrix(c(0, 1,
                2, 3,
                4, 5),
              nrow = 3, ncol = 2, byrow = TRUE)
x
x <- array_reshape(x, dim = c(6, 1))
x
x <- array_reshape(x, dim = c(3, 2))
x
x <- array_reshape(x, dim = c(2, 3))
x

# special case of reshaping that's commonly encountered is transposition.
# Transposing a matrix means exchanging its rows and its columns,
# so that x[i,] becomes x[, i]. The t() function can be used to transpose a matrix:

x <- matrix(1, nrow = 3, ncol = 2)
x
dim(x)


x <- t(x)
x
dim(x)
