#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
##################### Libraries#################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
library(ranger)
library(xgboost)
library(adabag)
library(dplyr)
library(torch)
library(torchvision)
library(ParBayesianOptimization)
library(SFDesign)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
##################### Load data and clean data #################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

### Bank
bank <- read.csv2("C:/Users/Karan/Desktop/HPO/data/bank/bank-full.csv", stringsAsFactors = TRUE)

### Heart
heart_path <- "C:/Users/Karan/Desktop/HPO/data/heart/processed.cleveland.data"
heart <- read.csv(heart_path, header = FALSE, na.strings = c("?", "-9"))
heart <- na.omit(heart)
colnames(heart) <- c(
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
)

heart$y <- as.factor(ifelse(heart$num == 0, 0, 1))

### Higgs
#higgs <- read.csv("C:/Users/Karan/Desktop/HPO/data/higgs/HIGGS.csv", nrows = 100000)
#higgs$X1.000000000000000000e.00 <- as.factor(higgs$X1.000000000000000000e.00)


# Forest Fire
fire <- read.csv("C:/Users/Karan/Desktop/HPO/data/forest fire/forestfires.csv")
fire$month <- as.factor(fire$month)
fire$day <- as.factor(fire$day)

# Superconductivity
super <- read.csv("C:/Users/Karan/Desktop/HPO/data/Superconductivity/train.csv")

# Bike
bike <- read.csv("C:/Users/Karan/Desktop/HPO/data/bike/hour.csv")
bike$day <- as.numeric(format(as.Date(bike$dteday), "%d"))
bike$dteday <- NULL
bike$casual <- NULL
bike$registered <- NULL

# Image Dataset Loaders
# MNIST

mnist_train <- mnist_dataset(
  root = "C:/Users/Karan/Desktop/HPO/data/MNIST",
  train = TRUE, download = TRUE,
  transform = function(x) {
    x <- torch_tensor(x, dtype = torch_float()) / 255  # convert to tensor + scale
    x$unsqueeze(1)  # add channel dimension (NCHW format -> 1 x 28 x 28)
  }
)
mnist_test <- mnist_dataset(
  root = "C:/Users/Karan/Desktop/HPO/data/MNIST",
  train = FALSE, download = TRUE,
  transform = function(x) {
    x <- torch_tensor(x, dtype = torch_float()) / 255
    x$unsqueeze(1)
  }
)

# CIFAR 10 / 100
cifar10_train <- cifar10_dataset(
  root = "C:/Users/Karan/Desktop/HPO/data/CIFAR10",
  train = TRUE, download = TRUE,
  transform = function(x) {
    x <- torch_tensor(x, dtype = torch_float()) / 255  # 0-1
    x$permute(c(3, 1, 2))  # HWC -> CHW
  }
)
cifar10_test <- cifar10_dataset(
  root = "C:/Users/Karan/Desktop/HPO/data/CIFAR10",
  train = FALSE, download = TRUE,
  transform = function(x) {
    x <- torch_tensor(x, dtype = torch_float()) / 255
    x$permute(c(3, 1, 2))
  }
)
cifar100_train <- cifar100_dataset(
  root = "C:/Users/Karan/Desktop/HPO/data/CIFAR10",
  train = TRUE, download = TRUE,
  transform = function(x) {
    x <- torch_tensor(x, dtype = torch_float()) / 255
    x$permute(c(3, 1, 2))
  }
)
cifar100_test <- cifar100_dataset(
  root = "C:/Users/Karan/Desktop/HPO/data/CIFAR10",
  train = FALSE, download = TRUE,
  transform = function(x) {
    x <- torch_tensor(x, dtype = torch_float()) / 255
    x$permute(c(3, 1, 2))
  }
)

# Train / Validation Set Splitter
# Given a dataset length n, it will return the train / valid split indices
split_train_valid <- function(n, train_frac = 0.8) {
  
  idx <- sample(1:n)
  
  n_train <- floor(n * train_frac)
  n_val   <- n - n_train
  
  list(
   train_idx = idx[1:n_train],
   val_idx   = idx[(n_train + 1):n]
  )
}


# Root Mean Squared Error for regression test error
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
###################### Objective Functions #####################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Given a dataset (variates X and target y) and a set of hyper-parameters
# these functions will fit the model and return a validation set loss

# Random Forest #

# params is a list with elements: m & depth
obj_RF <- function(X, y, params, type="reg", r_frac = 1, mode="train", X_test=NULL, y_test=NULL) {
  
  # data
  if(type == "reg") {
    Dtrain <- data.frame(X, y = y)
  } else {
    Dtrain <- data.frame(X, y = as.factor(y))
  }
  
  
  
  if(mode == "test") {
    # Fit model
    model <- ranger::ranger(
      formula = y ~ .,
      data = Dtrain,
      num.trees = 500,
      mtry = max(1, round(ncol(X) * params$m)), # convert fraction to int
      max.depth = params$depth      
    )
    
    

    # predict
    preds <- predict(model, data=data.frame(X_test))
    
    # get error
    if(type == "reg") {
      return(rmse(y_test, preds$predictions))
    } else {
      return(mean(preds$predictions != y_test))
    }
    
  }
  
  
  
  # use a subset of the data
  if(r_frac != 1) {
    if(type == "reg") {
      n_rows <- max(1, floor(nrow(Dtrain) * r_frac))
      idx <- sample(1:nrow(Dtrain), n_rows)
      Dtrain <- Dtrain[idx, , drop = FALSE]
    } else {
      idx_0 <- sample(which(Dtrain$y == "0"), max(1, floor(sum(Dtrain$y == "0") * r_frac)))
      idx_1 <- sample(which(Dtrain$y == "1"), max(1, floor(sum(Dtrain$y == "1") * r_frac)))
      idx <- c(idx_0, idx_1)
      Dtrain <- Dtrain[idx, , drop = FALSE]
    }
    
  }
  
  # Fit model
  model <- ranger::ranger(
    formula = y ~ .,
    data = Dtrain,
    num.trees = 500,
    mtry = max(1, round(ncol(X) * params$m)), # convert fraction to int
    max.depth = params$depth,
    oob.error = TRUE       
  )
  
  # Return OOB error
  if(type == "reg") {
    sqrt(model$prediction.error) # change to rmse instead of mse
  } else {
    model$prediction.error # return misclassification error as is
  }
}

# XGBoost # 

# params is a list with elements: depth, eta, subsample, colsample

obj_XGB <- function(X, y, params, type="reg", r_frac = 1, mode="train", X_test=NULL, y_test=NULL) {
  
  # Ready the data
  if (type == "class") { # xgb does not want factors, but numeric
    y <- as.numeric(as.character(y))  # ensure 0/1 numeric from the start
  }
  
  # Optionally use a subset
  if (r_frac != 1) {
    
    if(type == "reg") {
      n_rows <- max(1, floor(nrow(X) * r_frac))
      idx <- sample(1:nrow(X), n_rows)
    } else {
      idx_0 <- sample(which(y == "0"), max(1, floor(sum(y == "0") * r_frac)))
      idx_1 <- sample(which(y == "1"), max(1, floor(sum(y == "1") * r_frac)))
      idx <- c(idx_0, idx_1)
    }
    
    X_sub <- X[idx, , drop = FALSE]
    y_sub <- y[idx]
    
  } else {
    X_sub <- X
    y_sub <- y
  }
  
  

  
  Dtrain <- xgboost::xgb.DMatrix(data = model.matrix(~ . - 1, data=X_sub), label = y_sub)
  
  
  
  
  # Run CV
  res <- xgboost::xgb.cv(
    objective = ifelse(type == "reg", "reg:squarederror", "binary:logistic"),  
    eval_metric = ifelse(type == "reg", "rmse", "error"),           
    max_depth = params$depth,
    eta = params$eta,
    subsample = params$subsample, 
    colsample_bytree = params$colsample,
    data = Dtrain,
    nrounds = 3000, # arbitrarily large, we expect to stop early
    nfold = 3,  
    stratified = (type != "reg"),
    verbose = 0,
    early_stopping_rounds = 20 # stop if no improvement after 20 rounds
  )
  
  # Do test error instead if in test mode
  if(mode == "test") {
    model <- xgboost::xgb.train(
      objective = ifelse(type == "reg", "reg:squarederror", "binary:logistic"),  
      eval_metric = ifelse(type == "reg", "rmse", "error"),           
      max_depth = params$depth,
      eta = params$eta,
      subsample = params$subsample, 
      colsample_bytree = params$colsample,
      data = Dtrain,
      nrounds = res$best_iteration,
      verbose = 0,
    )
    
    # predict
    Dtest <- xgboost::xgb.DMatrix(data = model.matrix(~ . - 1, data=X_test))
    preds <- predict(model, newdata=Dtest)
    
    # get error
    if(type == "reg") {
      return(rmse(y_test, preds))
    } else {
      return(mean(as.numeric(preds > 0.5) != y_test))
    }
    
  } else { # return cv error
    # Return Results
    if (type == "reg") {
      return(min(res$evaluation_log$test_rmse_mean))
    } else {
      return(min(res$evaluation_log$test_error_mean))
    }
  }
}


# AdaBoost #

# params is a list with elements: B (number of trees) & depth (max depth in a tree)
obj_ADA <- function(X, y, params, type="class", r_frac = 1, mode="train", X_test=NULL, y_test=NULL) {
  
  # data
  Dtrain <- data.frame(X, y = as.factor(y))
  
  
  
  if(mode == "test") {
    # Fit model
    model <- adabag::boosting(
      formula = y ~ .,
      data = Dtrain,        
      mfinal = params$B, # number of boosting iterations
      control = rpart::rpart.control(maxdepth = params$depth),
    )
    
    
    # predict
    preds <- predict(model, newdata=data.frame(X_test))
    
    # get error
    return(mean(preds$class != y_test))
  }
  
  # use a subset of the data
  if(r_frac != 1) {
    idx_0 <- sample(which(Dtrain$y == "0"), max(1, floor(sum(Dtrain$y == "0") * r_frac)))
    idx_1 <- sample(which(Dtrain$y == "1"), max(1, floor(sum(Dtrain$y == "1") * r_frac)))
    idx <- c(idx_0, idx_1)
    Dtrain <- Dtrain[idx, , drop = FALSE]
  }
  
  # Fit model
  res <- adabag::boosting.cv(
    formula = y ~ .,
    data = Dtrain,              
    v = 5, # 5-fold CV
    mfinal = params$B, # number of boosting iterations
    control = rpart::rpart.control(maxdepth = params$depth),
    par = TRUE # run CV in parallel
  )
  
  # Return CV error
  res$error
}

## Feed Forward Neural Network


# requires that categoricals are mapped to ints or one-hot encoded before
## Hyper params:
## lr (learning rate)
## epochs
## batch_size
## weight_decay
## gamma (scheduler decay)
## Model Architecture defined as:
### params <- list(...
#                  size1 = integer, output size of linear layer
#                  act1 = 1,2,3 is code for act function
#                  size2 = int,
#                  act2 = 1:3,
#                  drop2 = real between 0 and 1, adds dropout after act2
#                  ...
### )
# final linear layer down to output size 1 (or 2 for class) is always included
# Activation functions:
## 0 = none, 1 = relu, 2 = tanh, 3 = sigmoid

# Maps int to activation function
get_act <- function(code) {
  if (code == 1) {
    return(nn_relu())
  } else if (code == 2) {
    return(nn_tanh())
  } else if (code == 3) {
    return(nn_sigmoid())  
  } else {
    return(nn_identity())  
  }
}

# Forces data to be tensors, does not one-hot encode anything, it forces numeric
prepare_tensors <- function(X, y, type = c("reg", "class")) {
  type <- match.arg(type)
  
  # --- X preprocessing ---
  X <- as.data.frame(X)
  X[] <- lapply(X, function(col) {
    if (is.factor(col) || is.character(col)) {
      as.numeric(as.factor(col))  # integer encode categoricals
    } else {
      as.numeric(col)             # force numeric
    }
  })
  
  X <- as.matrix(X)
  X <- torch_tensor(X, dtype = torch_float())
  
  # --- y preprocessing ---
  if (type == "reg") {
    y <- as.numeric(y)
    y <- torch_tensor(y, dtype = torch_float())
    y <- y$unsqueeze(2)   # make it (n,1) instead of (n)
  } else {
    # classification: ensure integer labels start at 1
    if (is.factor(y) || is.character(y)) {
      y <- as.integer(as.factor(y))  # R indices start at 1
    } else {
      y <- as.integer(y)
      if (min(y) == 0) y <- y + 1    # shift 0-based to 1-based
    }
    y <- torch_tensor(y, dtype = torch_long())
  }
  
  list(X = X, y = y)
}



# Feed Forward Neural Network objective function

obj_FFNN <- function(X, y, params, type="reg", r_frac = 1, mode="train", X_test=NULL, y_test=NULL) {
  # Format data ########################
  if(mode == "train") {
    data_prep <- prepare_tensors(X, y, type = type)
    X <- data_prep$X
    y <- data_prep$y
    
    n <- X$size(1)
    p <- X$size(2)
    train_idx <- sample(n, size = floor(0.8 * n))
    
    X_train <- X[train_idx, ]
    y_train <- y[train_idx]
    
    X_valid <- X[-train_idx, ]
    y_valid <- y[-train_idx]
    
    train_dl <- dataloader(
      tensor_dataset(X_train, y_train),
      batch_size = params$batch_size,
      shuffle = TRUE
    )
  } else {
    # Train on all of X, y
    data_prep <- prepare_tensors(X, y, type = type)
    X_train <- data_prep$X
    y_train <- data_prep$y
    
    n <- X_train$size(1)
    p <- X_train$size(2)
    
    
    train_dl <- dataloader(
      tensor_dataset(X_train, y_train),
      batch_size = params$batch_size,
      shuffle = TRUE
    )
    
    test_prep <- prepare_tensors(X_test, y_test, type = type)
    Xtest <- test_prep$X
    ytest <- test_prep$y
  }

  # Define Model ################################
  
  ffnet <- nn_module(
    initialize = function(input_size, params, output_size = 1) {
      self$layers <- nn_module_list()
      
      prev_size <- input_size
      i <- 1
      
      while(TRUE) {
        size_key <- paste0("size", i)
        act_key <- paste0("act", i)
        drop_key <- paste0("drop", i)
        
        # exit once all layers have been added
        if(!size_key %in% names(params)) break
        
        # Skip layer if size is 0
        if(params[[size_key]] == 0) {
          i <- i + 1
          next
        }  
        
        # add linear layer
        self$layers$append(nn_linear(prev_size, params[[size_key]]))
        
        # Activation function
        self$layers$append(get_act(params[[act_key]]))
        
        # Add dropout if it exists
        if(drop_key %in% names(params)) {
          self$layers$append(nn_dropout(p = params[[drop_key]]))
        } 
        
        prev_size <- params[[size_key]]
        i <- i + 1
      }
      
      # Final layer
      self$layers$append(nn_linear(prev_size, output_size))
    },
    
    forward = function(x) {
      len <- length(self$layers)
      for (i in 1:len) {
        x <- self$layers[[i]](x)
      }
      x
    }
  )
  
  # Initialize the model
  model <- ffnet(
    input_size = p,
    params = params,
    output_size = ifelse(type == "reg", 1,
                         length(unique(as.integer(y))))
  )
  
  ## Training loop #######################
  
  # Loss and optimizer
  if (type == "reg") {
    loss_func <- nn_mse_loss()
  } else {
    loss_func <- nn_cross_entropy_loss()
  }

  optimizer <- optim_adam(model$parameters, 
                          lr = params$lr, 
                          weight_decay=params$weight_decay)
  
  scheduler <- lr_step(optimizer, step_size = 1, gamma = params$gamma)
  
  # Training loop
  # This is for Hyperband purposes, uses only a fraction of the batches
  num_batches <- length(train_dl)
  use_batches <- ceiling(num_batches * r_frac)
  batch_count <- 0
  
  for (epoch in 1:params$epochs) {
    
    coro::loop(for (batch in train_dl) {
      
      batch_count <- batch_count + 1
      if (batch_count > use_batches) break

      inputs <- batch[[1]]
      targets <- batch[[2]]
      
      optimizer$zero_grad()
      
      outputs <- model(inputs)
      loss <- loss_func(outputs, targets)
      
      loss$backward()
      optimizer$step()
      
    })
    
    scheduler$step()
  }

  ## Return test error
  model$eval()
  
  
  if(mode == "train") {
    with_no_grad({
      if(type == "reg"){
        val_loss <- torch_sqrt(loss_func(model(X_valid), y_valid))
      } else {
        logits <- model(X_valid)                  
        preds  <- torch_argmax(logits, dim = 2) 
        val_loss <- (preds != y_valid)$to(dtype = torch_float())$mean()
      }
      
    })
    return(as.numeric(val_loss))
  } else {
    with_no_grad({
      if(type == "reg"){
        test_error <- loss_func(model(Xtest), ytest)
      } else {
        logits <- model(Xtest)                  
        preds  <- torch_argmax(logits, dim = 2) 
        test_error <- (preds != ytest)$to(dtype = torch_float())$mean()
      }
    })
    return(as.numeric(test_error))
  }
}

## Convolutional Neural Network ###


# Takes the name of some image dataset with a binary classification task
# Dataset that can be used are: MNIST, CIFAR10 & CIFAR100
# CNN architecture is defined as follows:
# Conv block:
#    

obj_CNN <- function(set_name = "MNIST", params, type="class", r_frac = 1, mode="train"){
  # Get data
  if(mode == "train") {
    if(set_name == "MNIST") {
      n <- length(mnist_train)
      splits <- split_train_valid(n)
      
      train_dl <- dataloader(dataset_subset(mnist_train, indices = splits$train_idx), 
                             batch_size = params$batch_size, shuffle = TRUE)
      valid_dl <- dataloader(dataset_subset(mnist_train, indices = splits$val_idx), 
                             batch_size = 128)
      num_classes <- 10
      in_channels <- 1
      
    } else if(set_name == "CIFAR10") {
      n <- length(cifar10_train)
      splits <- split_train_valid(n)
      
      train_dl <- dataloader(dataset_subset(cifar10_train, indices = splits$train_idx), 
                             batch_size = params$batch_size, shuffle = TRUE)
      valid_dl <- dataloader(dataset_subset(cifar10_train, indices = splits$val_idx), 
                             batch_size = 128)
      num_classes <- 10
      in_channels <- 3
      
    } else { # CIFAR 100
      n <- length(cifar100_train)
      splits <- split_train_valid(n)
      
      train_dl <- dataloader(dataset_subset(cifar100_train, indices = splits$train_idx), 
                             batch_size = params$batch_size, shuffle = TRUE)
      valid_dl <- dataloader(dataset_subset(cifar100_train, indices = splits$val_idx), 
                             batch_size = 128)
      num_classes <- 100
      in_channels <- 3
    }
  } else {
    if(set_name == "MNIST") {
      
      train_dl <- dataloader(mnist_train, batch_size = params$batch_size, shuffle = TRUE)
      test_dl <- dataloader(mnist_test, batch_size = 128)
      
      num_classes <- 10
      in_channels <- 1
      
    } else if(set_name == "CIFAR10") {

      train_dl <- dataloader(cifar10_train, batch_size = params$batch_size, shuffle = TRUE)
      test_dl <- dataloader(cifar10_test, batch_size = 128)
      
      num_classes <- 10
      in_channels <- 3
      
    } else { # CIFAR 100

      train_dl <- dataloader(cifar100_train, batch_size = params$batch_size, shuffle = TRUE)
      test_dl <- dataloader(cifar100_test, batch_size = 128)
      
      num_classes <- 100
      in_channels <- 3
    }
  }
  
  
  # Define Model ################################

  CNN <- nn_module(
    initialize = function(input_channels, params, output_size = 1) {
      self$layers <- nn_module_list()
      
      prev_channels <- input_channels
      i <- 1
      
      while(TRUE) {
        conv_key <- paste0("conv", i)
        if (!conv_key %in% names(params)) break  # stop when no more conv layers
        
        out_channels <- params[[conv_key]]
        kernel_size  <- ifelse(paste0("kernel", i) %in% names(params), params[[paste0("kernel", i)]], 3)
        drop_val     <- ifelse(paste0("drop", i) %in% names(params), params[[paste0("drop", i)]], 0)
        pool_val     <- ifelse(paste0("pool", i) %in% names(params), params[[paste0("pool", i)]], 0)
        
        # Skip layer if size is 0
        if(params[[conv_key]] == 0) {
          i <- i + 1
          next
        }  
        
        # Always include conv2D and ReLU
        self$layers$append(
          nn_conv2d(prev_channels, out_channels, kernel_size, 
                    stride = 1, padding = floor(kernel_size/2))
        )
        self$layers$append(nn_relu())
        
        # Optional dropout
        if(drop_val > 0) {
          self$layers$append(nn_dropout2d(p = drop_val))
        }
        # Optional max pooling
        if(pool_val > 0 ) {
          self$layers$append(nn_max_pool2d(kernel_size = pool_val))
        }
        
        prev_channels <- out_channels
        i <- i + 1
      }
      
      # GAP, Flatten and final linear layer
      self$layers$append(nn_adaptive_avg_pool2d(output_size = c(1,1)))
      self$layers$append(nn_flatten(start_dim = 2))
      self$layers$append(nn_linear(prev_channels, output_size))
    },
    
    forward = function(x) {
      len <- length(self$layers)
      for (i in 1:len) {
        x <- self$layers[[i]](x)
      }
      x
    }
  )
  

  # Initialize the model
  model <- CNN(
    input_channels = in_channels,
    params = params,
    output_size = num_classes
  )
  
  
  
  ## Training loop #######################
  
  # Loss and optimizer
  loss_func <- nn_cross_entropy_loss()

  optimizer <- optim_adam(model$parameters, 
                          lr = params$lr, 
                          weight_decay=params$weight_decay)
  
  scheduler <- lr_step(optimizer, step_size = 1, gamma = params$gamma)
  
  # Training loop
  # This is for Hyperband purposes, uses only a fraction of the batches
  num_batches <- length(train_dl)
  use_batches <- ceiling(num_batches * r_frac)
  batch_count <- 0
  
  for (epoch in 1:params$epochs) {
    
    coro::loop(for (batch in train_dl) {
      
      batch_count <- batch_count + 1
      if (batch_count > use_batches) break
      
      #inputs <- batch[[1]]
      #targets <- batch[[2]]
      
      inputs <- batch[[1]]$to(dtype = torch_float())
      targets <- batch[[2]]$to(dtype = torch_long())  # keep labels as long for cross_entropy
      
      
      optimizer$zero_grad()
      
      outputs <- model(inputs)
      loss <- loss_func(outputs, targets)
      
      loss$backward()
      optimizer$step()
      
    })
    
    scheduler$step()
  }

  ## Return test error
  model$eval()
  
  if(mode == "train") {
    valid_loss <- numeric()
    
    with_no_grad({
      coro::loop(for (batch in valid_dl) {
        # Each batch is a list: batch[[1]] = images, batch[[2]] = labels
        output <- model(batch[[1]])
        loss <- loss_func(output, batch[[2]])
        valid_loss <- c(valid_loss, as.numeric(loss))
      })
    })
    
    return(mean(valid_loss))
  } else {
    with_no_grad({
      total_err <- c()
      coro::loop(for (batch in test_dl) {

        logits <- model(batch[[1]])
        preds <- torch_max(logits, dim = 2)[[2]]
        
        batch_err <- (preds != batch[[2]])$to(dtype = torch_float())$mean()
        total_err <- c(total_err, as.numeric(batch_err))
      })
      
      return(mean(total_err))
    })
  }
}


################## Optimization Functions ######################


# Grid Search # 
# Will evaluate every hyper-parameter combination 
# param_values is a list where each element is a vector of hyper-parameter
# values. Every combination is tried. 
grid_search <- function(X=NULL, y=NULL, setname = NULL, obj_func, param_values=NULL, gd=NULL, type="class", X_test=NULL, y_test=NULL) {
  
  scoreFunction <- function(p){
    if(is.null(setname)) {
      obj_func(X=X, y=y, params=p, type=type)
    } else {
      obj_func(set_name=setname, params=p, type=type)
    }
  }
  
  t1 <- Sys.time()
  
  if(is.null(gd)) {
    # Turn param_values into a grid with every combination
    grid <- expand.grid(param_values, stringsAsFactors = FALSE)
  } else {
    grid <- gd # use pre-specified grid
  }
  
  
  # Evaluate objective function at each point
  n <- nrow(grid)
  errors <- numeric(length = n)
  for(i in 1:n) {
    errors[i] <- scoreFunction(grid[i, ])
    cat(sprintf("\rGrid search: %d / %d", i, n))
  }
  
  # find best model (lowest error)
  best_idx <- which.min(errors)
  best <- grid[best_idx, ]
  
  # Return results
  t_final <- as.numeric(Sys.time() - t1, units = "secs")
    
  # Get test error
  if(is.null(setname)){
    test_error <- obj_func(X=X,y=y, params=best, type=type, r_frac=1, mode="test", X_test=X_test, y_test=y_test)
  } else {
    test_error <- obj_func(set_name=setname, params=best, type=type, r_frac=1, mode="test")
  }
  
  
  
  if(is.null(setname)){
    #dlab <- sub("^X_", "", deparse(substitute(X)))
    dlab <- sub("^X_([^\\[]*).*", "\\1", deparse(substitute(X)))
  } else {
    dlab <- setname
  }
  
  list(opt = "Grid", 
       model = sub("^obj_", "", deparse(substitute(obj_func))),
       dataset  = dlab,
       type = type,
       val_error = format(round(errors[best_idx], 5), scientific = FALSE),
       test_error = round(test_error, 5),
       runtime = t_final,
       n_models = n,
       params = best)
}


# Random Search #

# param_dists contains the distribution function for each parameter
# where given the number of obs to create it will output that many obs
# Example input:
# param_dists <- list(
#   mtry          = function(n) runif(n, 2, 6),         # uniform [2,6]
#   min.node.size = function(n) sample(1:5, n, TRUE)    # integers 1–5
# )

random_search <- function(X=NULL, y=NULL, setname = NULL, obj_func, param_dists, n, type="class", X_test=NULL, y_test=NULL) {
  
  scoreFunction <- function(p){
    if(is.null(setname)) {
      obj_func(X=X, y=y, params=p, type=type)
    } else {
      obj_func(set_name=setname, params=p, type=type)
    }
  }

  
  
  t1 <- Sys.time()
  # get n many realizations of the parameters
  grid <- data.frame(row.names = 1:n)
  for(p in names(param_dists)) {
    grid[[p]] <- param_dists[[p]](n)
  }
  
  
  # Evaluate objective function at each point
  errors <- numeric(length = n)
  for(i in 1:n) {
    errors[i] <- scoreFunction(grid[i, ])
    cat(sprintf("\rRandom search: %d / %d", i, n))
  }
  
  # find best model (lowest error)
  best_idx <- which.min(errors)
  best <- grid[best_idx, ]
  
  # Return results
  t_final <- as.numeric(Sys.time() - t1, units = "secs")
  
  # Get test error
  if(is.null(setname)){
    test_error <- obj_func(X=X,y=y, params=best, type=type, r_frac=1, mode="test", X_test=X_test, y_test=y_test)
  } else {
    test_error <- obj_func(set_name=setname, params=best, type=type, r_frac=1, mode="test")
  }
  
  if(is.null(setname)){
    dlab <- sub("^X_([^\\[]*).*", "\\1", deparse(substitute(X)))
  } else {
    dlab <- setname
  }
  
  list(opt = "Random", 
       model = sub("^obj_", "", deparse(substitute(obj_func))),
       dataset  = dlab,
       type = type,
       val_error = format(round(errors[best_idx], 5), scientific = FALSE),
       test_error = round(test_error, 5),
       runtime = t_final,
       n_models = n,
       params = best)
}

# Bayesian Optimization #


bayes_search <- function(X=NULL, y=NULL, setname = NULL, obj_func, param_bounds, n=12, type="class", X_test=NULL, y_test=NULL, t_limit=NULL) {
  
  
  scoreFunction <- function(...){
    # negative since bayesOpt is a maximizer
    if(is.null(setname)) {
      list(Score = -obj_func(X=X, y=y, params=list(...), type=type))
    } else {
      list(Score = -obj_func(set_name=setname, params=list(...), type=type))
    }
    
  }
  
  #init_points <- max(6, floor(length(param_bounds)*1.5))
  init_points <- max(6, length(param_bounds)+1)
  
  t1 <- Sys.time()
  
  optObj <- ParBayesianOptimization::bayesOpt(
    FUN = scoreFunction,
    bounds = param_bounds,
    initPoints = init_points, # randomly eval at some points initially
    iters.n = n,
    iters.k = 1,
    otherHalting = list(timeLimit = t_limit)
  )
  
  # Get best params
  best <- getBestPars(optObj)
  best_error <- -max(optObj$scoreSummary$Score)
  
  t_final <- as.numeric(Sys.time() - t1, units = "secs")
  
  # Get test error
  if(is.null(setname)){
    test_error <- obj_func(X=X,y=y, params=best, type=type, r_frac=1, mode="test", X_test=X_test, y_test=y_test)
  } else {
    test_error <- obj_func(set_name=setname, params=best, type=type, r_frac=1, mode="test")
  }
  
  if(is.null(setname)){
    #dlab <- sub("^X_", "", deparse(substitute(X)))
    dlab <- sub("^X_([^\\[]*).*", "\\1", deparse(substitute(X)))
  } else {
    dlab <- setname
  }
  
  # Return Results
  list(opt = "Bayes", 
       model = sub("^obj_", "", deparse(substitute(obj_func))),
       dataset  = dlab,
       type = type,
       val_error = format(round(best_error, 5), scientific = FALSE),
       test_error = round(test_error, 5),
       runtime = t_final,
       n_models = nrow(optObj$scoreSummary),
       params = data.frame(getBestPars(optObj)))
}


# Hyper-band with random search#

hyperband <- function(X=NULL, y=NULL, setname = NULL, obj_func, param_dists, R, eta = 3, type="class", X_test=NULL, y_test=NULL) {
  score_func <- function(r_frac, params) {
    if(!is.null(setname)) {
      obj_func(setname, params, type, r_frac)
    } else {
      obj_func(X, y, params, type, r_frac)
    }
  }
  
  all_results <- list()
  
  t1 <- Sys.time()
  
  n_models <- 0 
  
  # maximum number of brackets
  s_max <- floor(log(R, base = eta))
  B <- (s_max + 1) * R
  
  for(s in s_max:0) {
    
    n <- ceiling((B / R) * eta^s / (s + 1))
    r <- R * eta^(-s)
    
    # Begin Successive Halving in inner loop
    # get n many realizations of the parameters
    grid <- data.frame(row.names = 1:n)
    for(p in names(param_dists)) {
      grid[[p]] <- param_dists[[p]](n)
    }
    
    n_models <- n_models + n
    
    for(i in 0:s) {
      n_i <- nrow(grid)
      r_i <- r * eta^i
      
      # Get losses
      L <- numeric(length = n_i)
      for(j in 1:n_i) {
        # r_frac is normalized resource allocation
        L[j] <- score_func(r_frac = r_i/R, params = grid[j, ])
        cat(sprintf("\rHyperband. Brackets %d / %d. Successive Halving Progress: %d / %d. Round Progress: %d / %d.", s_max+1-s, s_max+1, i, s, j, n_i))
      }
      
      grid$error <- L
      all_results[[length(all_results) + 1]] <- grid
      
      
      # Update grid to be just the top n_i / eta performers
      k <- max(1, floor(n_i / eta))
      grid <- head(grid[order(grid$error), ], k)
    }
  }
  
  # Combine everything
  results <- do.call(rbind, all_results)
  
  # find best model (lowest error)
  best_idx <- which.min(results$error)
  best <- results[best_idx, ]
  best$error <- NULL
  
  t_final <- as.numeric(Sys.time() - t1, units = "secs")
  
  # Get test error
  if(is.null(setname)){
    test_error <- obj_func(X=X,y=y, params=best, type=type, r_frac=1, mode="test", X_test=X_test, y_test=y_test)
  } else {
    test_error <- obj_func(set_name=setname, params=best, type=type, r_frac=1, mode="test")
  }
  
  if(is.null(setname)){
    #dlab <- sub("^X_", "", deparse(substitute(X)))
    dlab <- sub("^X_([^\\[]*).*", "\\1", deparse(substitute(X)))
  } else {
    dlab <- setname
  }
  
  # Return results
  list(opt = "Hyperband", 
       model = sub("^obj_", "", deparse(substitute(obj_func))),
       dataset  = dlab,
       type = type,
       val_error = format(round(results$error[best_idx], 5), scientific = FALSE),
       test_error = round(test_error, 5),
       runtime = t_final,
       n_models = n_models,
       params = best)
}

hyperband_print <- function(R, eta = 3) {

  all_results <- list()
  bracket_table <- list()   # <- store info for the summary table
  

  # maximum number of brackets
  s_max <- floor(log(R, base = eta))
  B <- (s_max + 1) * R
  
  for(s in s_max:0) {
    
    n <- ceiling((B / R) * eta^s / (s + 1))
    r <- R * eta^(-s)
    
    # Begin Successive Halving in inner loop
    grid <- 1:n

    for(i in 0:s) {
      n_i <- floor(n * eta^(-i))

      r_i <- r * eta^i
      
      # store in bracket table
      bracket_table[[length(bracket_table)+1]] <- data.frame(
        s = s, i = i, n_i = n_i, r_i = r_i
      )
      
      all_results[[length(all_results) + 1]] <- grid
      

      k <- min(n_i, nrow(grid))   
      grid <- head(sample(grid), k)
    }
  }
  
  # Combine everything
  results <- do.call(rbind, all_results)
  table_data <- do.call(rbind, bracket_table)
  
  # Format table to wide form
  library(tidyr)
  library(dplyr)
  
  table_wide <- table_data %>%
    tidyr::pivot_wider(
      id_cols = i,
      names_from = s,
      values_from = c(n_i, r_i),
      names_glue = "s{s}_{.value}"
    ) %>%
    arrange(i)
  
  # Print table
  cat("\n=== Hyperband Bracket Summary ===\n")
  print(table_wide, row.names = FALSE)
  cat("=================================\n\n")
}

# Function which converts budget into number of objective function evals

hyperband_counter <- function(R, eta = 3) {

  count <- 0
  func_calls <- 0
  n_models <- 0
  

  # maximum number of brackets
  s_max <- floor(log(R, base = eta))
  B <- (s_max + 1) * R
  
  for(s in s_max:0) {
    
    n <- ceiling((B / R) * eta^s / (s + 1))
    r <- R * eta^(-s)
    
    # Begin Successive Halving in inner loop
    # get n many realizations of the parameters

    n_i <- n
    n_models <- n_models + n_i
    for(i in 0:s) {

      r_i <- r * eta^i
      
      for(j in 1:n_i) {
      
        count <- count + r_i/R
        func_calls <- func_calls + 1
      }
      
      n_i <- max(1, floor(n_i / eta))
    }
  }
  
  list(n_models = n_models,
       count = count,
       func_calls = func_calls)
}



plot(x=10:1000, y=as.vector(lapply(10:1000, FUN = function(x) {hyperband_counter(x)$func_calls})))

hyperband_counter(81, eta=3)

## Sequential Uniform Design Search

# AugmentUD takes existing points, and adds 'add'-many new points that
# minimize the wrap-around discrepency measure. Checks n_candidates-many 
# candidates each time a new point is added
augmentUD <- function(X0, add, n_candidates = 1000) {
  # X0: existing design (rows = runs, cols = factors), scaled to [0,1]
  # add: number of new points to add
  # n_candidates: number of candidate points to try each step
  
  d <- ncol(X0)
  new_points <- c()
  
  for (k in 1:add) {
    # generate random candidate points
    C <- matrix(runif(n_candidates * d), ncol = d)
    
    # score each candidate by discrepancy when added
    scores <- apply(C, 1, function(cand) {
      SFDesign::uniform.crit(rbind(X0, cand))
    })
    
    # pick best candidate
    best <- C[which.min(scores), , drop = FALSE]
    
    # update design
    X0 <- rbind(X0, best)
    new_points <- rbind(new_points, best)
    
  }
  
  new_points
}

seqUD <- function(X=NULL, y=NULL, setname = NULL, obj_func, param_bounds, T_max=3, n_points=20, type="class", X_test=NULL, y_test=NULL){
  
  scoreFunction <- function(p){
    if(is.null(setname)) {
      obj_func(X=X, y=y, params=p, type=type)
    } else {
      obj_func(set_name=setname, params=p, type=type)
    }
  }
  
  t1 <- Sys.time()
  
  d <- length(param_bounds)

  # Find range and min of each param

  mins <- sapply(param_bounds, function(x) as.numeric(x[1]))
  lwr <- mins
  upr <- sapply(param_bounds, function(x) as.numeric(x[2]))
  ranges <- sapply(param_bounds, function(x) as.numeric(diff(x)))
  is_int <- sapply(param_bounds, function(x) is.integer(x))
  
  from_01_to_ps <- function(U) {
    
    out <- matrix(NA, nrow = nrow(U), ncol = d)
    for (j in 1:d) {
      vals <- mins[j] + U[, j] * ranges[j]
      if (is_int[j]) {
        vals <- round(vals)
      }
      out[, j] <- vals
    }
    
    colnames(out) <- names(param_bounds)
    as.data.frame(out)
  }

  
  U01 <- c()
  grid <- c()
  box_radius <- 0.5
  
  for(t in 1:T_max) {
    # Generate points
    if(length(U01) == 0){ # Make fresh design
      new_points <- uniformLHD(n = n_points, p = d)$design
      # map to parameter space
      ps_points <- from_01_to_ps(new_points)
      
    } else { # Augment existing
      # update boundaries
      box_radius <- box_radius * 0.5
      lwr <- best - box_radius
      upr <- best + box_radius
      
      # Check if the box is outside [0,1]^d
      shift_lwr <- pmax(0 - lwr, 0)   # how far below 0
      shift_upr <- pmin(1 - upr, 0)   # how far above 1 (negative)
      
      # total shift = whichever adjustment is needed
      shift <- shift_lwr + shift_upr
      
      # apply shift to corners
      lwr <- lwr + shift
      upr <- upr + shift
   
      # Find which points are in the new bounds
      in_the_box <- apply(U01, MARGIN=1, FUN = function(x) {2 * d == sum(c(x > lwr, x < upr )  )})
      box_points <- U01[in_the_box, , drop=FALSE]
      
      # Augment the sub design
      n_e <- n_points - nrow(box_points)
      new_points <- augmentUD(sweep(box_points, 2, lwr, "-") / (box_radius * 2), add = n_e, n_candidates = 1000*d)
      
      # Shift the design back into place
      new_points <- (new_points * (box_radius * 2)) 
      new_points <- sweep(new_points, 2, lwr, "+")
      
      # Generate points in the parameter space
      ps_points <- from_01_to_ps(new_points)
    }
    
    # Evaluate each point

    n <- nrow(ps_points)
    errors <- numeric(length = n)
    
    for(i in 1:n) {
      errors[i] <- scoreFunction(ps_points[i, ])
      cat(sprintf("\rSeqUD: %d / %d. Iteration Progress: %d / %d.", t, T_max, i, n))
    }
    
    # Save the errors
    ps_points$error <- errors
    
    # Append new points onto old points
    grid <- rbind(grid, ps_points)
    U01 <- rbind(U01, new_points)
    
    # Find the best point
    best_idx <- which.min(grid$error)
    best <- U01[best_idx, ]
  }
  
  
  t_final <- as.numeric(Sys.time() - t1, units = "secs")
  
  best_params <- grid[best_idx, ]
  best_params$error <- NULL
  # Get test error
  if(is.null(setname)){
    test_error <- obj_func(X=X,y=y, params=best_params, type=type, r_frac=1, mode="test", X_test=X_test, y_test=y_test)
  } else {
    test_error <- obj_func(set_name=setname, params=best_params, type=type, r_frac=1, mode="test")
  }
  
  if(is.null(setname)){
    dlab <- sub("^X_([^\\[]*).*", "\\1", deparse(substitute(X)))
  } else {
    dlab <- setname
  }
  
  # Return results
  list(opt = "SeqUD", 
       model = sub("^obj_", "", deparse(substitute(obj_func))),
       dataset  = dlab,
       type = type,
       val_error = as.numeric(format(round(grid$error[best_idx], 5), scientific = FALSE)),
       test_error = round(test_error, 5),
       runtime = t_final,
       n_models = nrow(grid),
       params = best_params)
}


seqUD_counter <- function(d, n_points, T_max) {
  count <- n_points
  
  for(i in 2:T_max) {
    new_points <- n_points - round(n_points / (2^d))
    count <- count + new_points
  }

  count
}

seqUD_counter(4, 32, 3)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
################# Testing & Timing #############################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Load Data and Create Calibration (Train + Validation) / Test Split
X_bank <- subset(bank, select = -y)
y_bank <- bank$y
y_bank <- as.factor(ifelse(y_bank == "yes", 1, 0))
n_rows <- max(1, floor(nrow(X_bank) * 0.8))
bank_idx <- sample(1:nrow(X_bank), n_rows)

#X_higgs <- subset(higgs, select= -X1.000000000000000000e.00)
#y_higgs <- higgs$X1.000000000000000000e.00
#n_rows <- max(1, floor(nrow(X_higgs) * 0.8))
#higgs_idx <- sample(1:nrow(X_higgs), n_rows)

X_heart <- subset(heart, select = -c(y, num))
y_heart <- heart$y
n_rows <- max(1, floor(nrow(X_heart) * 0.8))
heart_idx <- sample(1:nrow(X_heart), n_rows)

X_fire <- subset(fire, select = -area)
y_fire <- fire$area
n_rows <- max(1, floor(nrow(X_fire) * 0.8))
fire_idx <- sample(1:nrow(X_fire), n_rows)

X_bike <- subset(bike, select= -cnt)
y_bike <- bike$cnt
n_rows <- max(1, floor(nrow(X_bike) * 0.8))
bike_idx <- sample(1:nrow(X_bike), n_rows)

X_super <- subset(super, select = -critical_temp)
y_super <- super$critical_temp
n_rows <- max(1, floor(nrow(X_super) * 0.8))
super_idx <- sample(1:nrow(X_super), n_rows)

################ Benchmarks ##############################


# Benchmark function
run_benchmarks <- function(params, dists, bounds, obj_func, n_comb, n_iter, budget, n_points, T_max, outputfile=NULL, include_reg=TRUE){
  
  res <- list()
  
  save_partial <- function(res_list, id) {
    if (!is.null(outputfile)) {
        temp_out <- paste0(sub("\\.csv$", "", outputfile)   , "_", id, ".csv")
        res_df_partial <- do.call(rbind, lapply(res_list, function(x) as.data.frame(x, row.names = NULL)))
        write.csv(res_df_partial, temp_out, row.names = FALSE)
        cat(paste0("[", Sys.time(), "] Progress saved to: ", temp_out, "\n"))
    }
  }
  
  
  # classification sets
  # grid search
  res[[1]] <- grid_search(X_bank[bank_idx,], y_bank[bank_idx], obj_func = obj_func, param_values = params, type="class", X_test = X_bank[-bank_idx,], y_test = y_bank[-bank_idx])
 # res[[2]] <- grid_search(X_higgs[higgs_idx,], y_higgs[higgs_idx], obj_func = obj_func, param_values = params, type="class", X_test = X_higgs[-higgs_idx,], y_test = y_higgs[-higgs_idx])
  res[[3]] <- grid_search(X_heart[heart_idx,], y_heart[heart_idx], obj_func = obj_func, param_values = params, type="class", X_test = X_heart[-heart_idx,], y_test = y_heart[-heart_idx])
  print("grid: class, done.")
  save_partial(res, 0)
  
  # Random search
  res[[4]] <- random_search(X_bank[bank_idx,], y_bank[bank_idx], obj_func = obj_func, param_dists = dists, type="class", n=n_comb, X_test = X_bank[-bank_idx,], y_test = y_bank[-bank_idx])
 # res[[5]] <- random_search(X_higgs[higgs_idx,], y_higgs[higgs_idx], obj_func = obj_func, param_dists = dists, type="class", n=n_comb, X_test = X_higgs[-higgs_idx,], y_test = y_higgs[-higgs_idx])
  res[[6]] <- random_search(X_heart[heart_idx,], y_heart[heart_idx], obj_func = obj_func, param_dists = dists, type="class", n=n_comb, X_test = X_heart[-heart_idx,], y_test = y_heart[-heart_idx])
  print("random: class, done.")
  save_partial(res, 1)
  
  # Bayes search
  res[[7]] <- bayes_search(X_bank[bank_idx,], y_bank[bank_idx], obj_func = obj_func, param_bounds = bounds, type="class", n=n_iter, X_test = X_bank[-bank_idx,], y_test = y_bank[-bank_idx])
 # res[[8]] <- bayes_search(X_higgs[higgs_idx,], y_higgs[higgs_idx], obj_func = obj_func, param_bounds = bounds, type="class", n=n_iter, X_test = X_higgs[-higgs_idx,], y_test = y_higgs[-higgs_idx])
  res[[9]] <- bayes_search(X_heart[heart_idx,], y_heart[heart_idx], obj_func = obj_func, param_bounds = bounds, type="class", n=n_iter, X_test = X_heart[-heart_idx,], y_test = y_heart[-heart_idx])
  print("bayes: class, done.")
  save_partial(res, 2)
  
  # Hyperband
  res[[10]] <- hyperband(X_bank[bank_idx,], y_bank[bank_idx], obj_func = obj_func, param_dists = dists, type="class", R = budget, X_test = X_bank[-bank_idx,], y_test = y_bank[-bank_idx])
 # res[[11]] <- hyperband(X_higgs[higgs_idx,], y_higgs[higgs_idx], obj_func = obj_func, param_dists = dists, type="class", R = budget, X_test = X_higgs[-higgs_idx,], y_test = y_higgs[-higgs_idx])
  res[[12]] <- hyperband(X_heart[heart_idx,], y_heart[heart_idx], obj_func = obj_func, param_dists = dists, type="class", R = budget, X_test = X_heart[-heart_idx,], y_test = y_heart[-heart_idx])
  print("hyperband: class, done.")
  save_partial(res, 3)
  
  # SeqUD 
  res[[13]] <- seqUD(X_bank[bank_idx,], y_bank[bank_idx], obj_func = obj_func, param_bounds = bounds, type="class", n_points=n_points, T_max = T_max, X_test = X_bank[-bank_idx,], y_test = y_bank[-bank_idx])
 # res[[14]] <- seqUD(X_higgs[higgs_idx,], y_higgs[higgs_idx], obj_func = obj_func, param_bounds = bounds, type="class", n_points=n_points, T_max = T_max, X_test = X_higgs[-higgs_idx,], y_test = y_higgs[-higgs_idx])
  res[[15]] <- seqUD(X_heart[heart_idx,], y_heart[heart_idx], obj_func = obj_func, param_bounds = bounds, type="class", n_points=n_points, T_max = T_max, X_test = X_heart[-heart_idx,], y_test = y_heart[-heart_idx])
  print("seqUD: class, done.")
  save_partial(res, 4)
  
  if(include_reg){ # regression sets
    
    # grid search
    res[[16]] <- grid_search(X_fire[fire_idx,], y_fire[fire_idx], obj_func = obj_func, param_values = params, type="reg", X_test = X_fire[-fire_idx,], y_test = y_fire[-fire_idx])
    res[[17]] <- grid_search(X_bike[bike_idx,], y_bike[bike_idx], obj_func = obj_func, param_values = params, type="reg", X_test = X_bike[-bike_idx,], y_test = y_bike[-bike_idx])
    res[[18]] <- grid_search(X_super[super_idx,], y_super[super_idx], obj_func = obj_func, param_values = params, type="reg", X_test = X_super[-super_idx,], y_test = y_super[-super_idx])
    print("grid: reg, done.")
    save_partial(res, 5)
    
    # Random search
    res[[19]] <- random_search(X_fire[fire_idx,], y_fire[fire_idx], obj_func = obj_func, param_dists = dists, type="reg", n=n_comb, X_test = X_fire[-fire_idx,], y_test = y_fire[-fire_idx])
    res[[20]] <- random_search(X_bike[bike_idx,], y_bike[bike_idx], obj_func = obj_func, param_dists = dists, type="reg", n=n_comb, X_test = X_bike[-bike_idx,], y_test = y_bike[-bike_idx])
    res[[21]] <- random_search(X_super[super_idx,], y_super[super_idx], obj_func = obj_func, param_dists = dists, type="reg", n=n_comb, X_test = X_super[-super_idx,], y_test = y_super[-super_idx])
    print("random: reg, done.")
    save_partial(res, 6)
    
    # Bayes search
    res[[22]] <- bayes_search(X_fire[fire_idx,], y_fire[fire_idx], obj_func = obj_func, param_bounds = bounds, type="reg", n=n_iter, X_test = X_fire[-fire_idx,], y_test = y_fire[-fire_idx])
    res[[23]] <- bayes_search(X_bike[bike_idx,], y_bike[bike_idx], obj_func = obj_func, param_bounds = bounds, type="reg", n=n_iter, X_test = X_bike[-bike_idx,], y_test = y_bike[-bike_idx])
    res[[24]] <- bayes_search(X_super[super_idx,], y_super[super_idx], obj_func = obj_func, param_bounds = bounds, type="reg", n=n_iter, X_test = X_super[-super_idx,], y_test = y_super[-super_idx])
    print("bayes: reg, done.")
    save_partial(res, 7)
    
    # Hyperband
    res[[25]] <- hyperband(X_fire[fire_idx,], y_fire[fire_idx], obj_func = obj_func, param_dists = dists, type="reg", R = budget, X_test = X_fire[-fire_idx,], y_test = y_fire[-fire_idx])
    res[[26]] <- hyperband(X_bike[bike_idx,], y_bike[bike_idx], obj_func = obj_func, param_dists = dists, type="reg", R = budget, X_test = X_bike[-bike_idx,], y_test = y_bike[-bike_idx])
    res[[27]] <- hyperband(X_super[super_idx,], y_super[super_idx], obj_func = obj_func, param_dists = dists, type="reg", R = budget, X_test = X_super[-super_idx,], y_test = y_super[-super_idx])
    print("hyperband: reg, done.")
    save_partial(res, 8)
    
    # SeqUD 
    res[[28]] <- seqUD(X_fire[fire_idx,], y_fire[fire_idx], obj_func = obj_func, param_bounds = bounds, type="reg", n_points=n_points, T_max = T_max, X_test = X_fire[-fire_idx,], y_test = y_fire[-fire_idx])
    res[[29]] <- seqUD(X_bike[bike_idx,], y_bike[bike_idx], obj_func = obj_func, param_bounds = bounds, type="reg", n_points=n_points, T_max = T_max, X_test = X_bike[-bike_idx,], y_test = y_bike[-bike_idx])
    res[[30]] <- seqUD(X_super[super_idx,], y_super[super_idx], obj_func = obj_func, param_bounds = bounds, type="reg", n_points=n_points, T_max = T_max, X_test = X_super[-super_idx,], y_test = y_super[-super_idx])
    print("seqUD: reg, done.")
    save_partial(res, 9)
  }
 
  
  res_df <- do.call(rbind, lapply(res, function(x) as.data.frame(x, row.names = NULL)))
  print(res_df)
  #as.data.frame(res[!names(res) %in% c("params")])
  if(!is.null(outputfile)){
    write.csv(res_df, outputfile, row.names = FALSE)
  }
  
}


#+++++++++++++++++++++++++++++++++#
########## Random Forest ##########
#+++++++++++++++++++++++++++++++++#

rf_params <- list(
  m = c(0.5, 0.6, 0.7, 0.8, 0.9, 1),
  depth = 1:6
)

rf_dists <- list(
  m = function(n) runif(n, 0.5, 1),         
  depth = function(n) sample(1:6, n, TRUE)    # integers 1–6
)

rf_bounds <- list(
  m = c(0.5,1),         
  depth = c(1L, 6L)
)

seqUD(X=X_heart[heart_idx, ], y=y_heart[heart_idx], obj_func = obj_RF, param_bounds = rf_bounds, n_points=20, T_max=2, X_test=X_heart[-heart_idx,], y_test = y_heart[-heart_idx])

rf_out <- "C:/Users/Karan/Desktop/HPO/results/rf.csv"
run_benchmarks(params=rf_params, dists=rf_dists, bounds=rf_bounds, obj_func=obj_RF, n_comb=36, n_iter=10, budget=225, n_points=18, T_max=2, outputfile=rf_out)




#+++++++++++++++++++++++++++++++++#
########## AdaBoost ###############
#+++++++++++++++++++++++++++++++++#

ADA_params <- list(
  B = c(50, 100, 200, 300, 500, 1000),
  depth = 1:6
)

ADA_dists <- list(
  B = function(n) sample(50:1000, n, TRUE),   # ints 50-1000      
  depth = function(n) sample(1:6, n, TRUE)    # integers 1–6
)

ADA_bounds <- list(
  B = c(50L, 1000L),         
  depth = c(1L, 6L)
)

ada_out <- "C:/Users/Karan/Desktop/HPO/results/ada.csv"
run_benchmarks(params=ADA_params, dists=ADA_dists, bounds=ADA_bounds, obj_func=obj_ADA, n_comb=36, n_iter=10, budget=225, n_points=18, T_max=2, outputfile=ada_out, include_reg = FALSE)

#+++++++++++++++++++++++++++++++++#
########## XGBoost ################
#+++++++++++++++++++++++++++++++++#


xgb_params <- list(
  eta = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
  depth = 1:6,
  subsample = c(0.7, 1),
  colsample = c(0.7, 1)
)

xgb_dists <- list(
  eta = function(n) pmax(pmin(runif(n), runif(n), runif(n)), 0.005), # min of 3 uniforms. favours smaller values        
  depth = function(n) sample(1:6, n, TRUE),  # Integer between 1 and 6 (inclusive)
  subsample = function(n) runif(n, 0.7, 1), # Uniform between 0.7 and 1
  colsample = function(n) runif(n, 0.7, 1) # Uniform between 0.7 and 1
)

xgb_bounds <- list(
  eta = c(0.001,0.5),         
  depth = c(1L, 6L),
  subsample = c(0.7, 1),
  colsample = c(0.7, 1)
)


xgb_out <- "C:/Users/Karan/Desktop/HPO/results/xgb.csv"
run_benchmarks(params=xgb_params, dists=xgb_dists, bounds=xgb_bounds, obj_func=obj_XGB, n_comb=144, n_iter=60, budget=1500, n_points=36, T_max=3, outputfile=xgb_out)


#+++++++++++++++++++++++++++++++++++++++++++++++++#
########## Feed Forward Neural Network ############
#+++++++++++++++++++++++++++++++++++++++++++++++++#

## Small datasets - Group A
params_A <- list(
  lr = 0.005,
  epochs = 200,
  batch_size = 16,
  weight_decay = 0.0005,
  gamma = 0.95,
  size1 = 32,  act1 = 1,
  size2 = 16,  act2 = 3, drop2 = 0.4
)

## Medium/Large datasets - Group B
params_B <- list(
  lr = 0.005,
  epochs = 125,
  batch_size = 128,
  weight_decay = 0.001,
  gamma = 0.90,
  size1 = 128, act1 = 1,
  size2 = 64,  act2 = 1, drop2 = 0.3,
  size3 = 32,  act3 = 2, drop3 = 0.2
)

## High Dimensional datasets - Group C
params_C <- list(
  lr = 0.003,
  epochs = 150,
  batch_size = 256,
  weight_decay = 0.0005,
  gamma = 0.92,
  size1 = 256, act1 = 1,
  size2 = 128, act2 = 2, drop2 = 0.3,
  size3 = 64,  act3 = 1, drop3 = 0.2
)

obj_FFNN(X_heart[heart_idx,], y_heart[heart_idx], params = params_A, type="class")
system.time(   )
system.time( obj_FFNN(X_fire[fire_idx,], y_fire[fire_idx], params = params_A, type="reg")  )
system.time( obj_FFNN(X_bank[bank_idx,], y_bank[bank_idx], params = params_B, type="class")  )
system.time( obj_FFNN(X_bike[bike_idx,], y_bike[bike_idx], params = params_B, type="reg")  )
system.time(obj_FFNN(X_super[super_idx,], y_super[super_idx], params = params_C, type="reg"))


ffnn_benchmark <- function(X, y, idx, grid, dists, bounds, n_comb, n_iter, budget, n_points, T_max, outputfile=NULL, type="reg"){
  res <- list()
  
  res[[1]] <- grid_search(X = X[idx, ], y=y[idx], obj_func = obj_FFNN, gd=grid, type=type, X_test=X[-idx,], y_test = y[-idx])
  res[[2]] <- random_search(X = X[idx, ], y=y[idx], obj_func = obj_FFNN, param_dists = dists, n=n_comb, type=type, X_test=X[-idx,], y_test = y[-idx])
  res[[3]] <- bayes_search(X = X[idx, ], y=y[idx], obj_func = obj_FFNN, param_bounds = bounds, n=n_iter, type=type, X_test=X[-idx,], y_test = y[-idx])
  res[[4]] <- hyperband(X = X[idx, ], y=y[idx], obj_func = obj_FFNN, param_dists = dists, R=budget, type=type, X_test=X[-idx,], y_test = y[-idx])
  res[[5]] <- seqUD(X = X[idx, ], y=y[idx], obj_func = obj_FFNN, param_bounds = bounds, n_points = n_points, T_max = T_max, type=type, X_test=X[-idx,], y_test = y[-idx])


  
  res_df <- do.call(rbind, lapply(res, function(x) as.data.frame(x, row.names = NULL)))
  print(res_df)

  if(!is.null(outputfile)){
    write.csv(res_df, outputfile, row.names = FALSE)
  }
}

## Small datasets - Group A
# 864 combs
ffnn_params_A <- list(
  lr = c(0.001, 0.003, 0.01),      
  epochs = c(500, 100),          
  batch_size = c(8, 16),
  weight_decay = c(0, 0.0005),         
  gamma = c(0.90, 0.99),
  
  size1 = c(16, 32),
  act1 = c(1),
  size2 = c(0, 8, 16),
  act2 = c(1, 2),                      
  drop2 = c(0.1, 0.3)
)

# make grid and remove dups
grid_A <- expand.grid(ffnn_params_A, stringsAsFactors = FALSE)
grid_A$drop2[grid_A$size2 == 0] <- NA
grid_A$act2[grid_A$size2 == 0]  <- NA

grid_A <- unique(grid_A)
nrow(grid_A)



ffnn_dists_A <- list(
  lr = function(n) 10^runif(n, -3, -2),           # ~0.001–0.01
  epochs = function(n) sample(50:100, n, TRUE),
  batch_size = function(n) sample(c(8, 16, 32), n, TRUE),
  weight_decay = function(n) runif(n, 0, 0.001),
  gamma = function(n) runif(n, 0.90, 0.99),
  
  size1 = function(n) sample(16:32, n, TRUE),
  act1 = function(n) rep(1, n),
  size2 = function(n) sample(0:16, n, TRUE),
  act2 = function(n) sample(c(1, 2, 3), n, TRUE),
  drop2 = function(n) runif(n, 0, 0.4)
)

ffnn_bounds_A <- list(
  lr = c(0.001, 0.01),
  epochs = c(50L, 100L),
  batch_size = c(8L, 32L),
  weight_decay = c(0, 0.001),
  gamma = c(0.90, 0.99),
  
  size1 = c(16L, 32L),
  act1 = c(1L, 2L),
  size2 = c(0L, 16L),
  act2 = c(1L, 3L),
  drop2 = c(0, 0.4)
)


out_ffnn_A <- "C:/Users/Karan/Desktop/HPO/results/ffnn_A.csv"

res_A <- list()

res_A[[1]] <- grid_search(X=X_heart[heart_idx,], y=y_heart[heart_idx], obj_func = obj_FFNN, gd = grid_A, type="class", X_test = X_heart[-heart_idx,], y_test =y_heart[-heart_idx])
res_A[[2]] <- random_search(X=X_heart[heart_idx,], y=y_heart[heart_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_A, n=864, type="class", X_test = X_heart[-heart_idx,], y_test =y_heart[-heart_idx])

res_A[[3]] <- grid_search(X=X_fire[fire_idx,], y=y_fire[fire_idx], obj_func = obj_FFNN, gd = grid_A, type="reg", X_test = X_fire[-fire_idx,], y_test =y_fire[-fire_idx])
res_A[[4]] <- random_search(X=X_fire[fire_idx,], y=y_fire[fire_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_A, n=864, type="reg", X_test = X_fire[-fire_idx,], y_test =y_fire[-fire_idx])

res_A[[5]] <- hyperband(X=X_heart[heart_idx,], y=y_heart[heart_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_A, R=9000, type="class", X_test = X_heart[-heart_idx,], y_test =y_heart[-heart_idx])
res_A[[6]] <- seqUD(X=X_heart[heart_idx,], y=y_heart[heart_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_A, n_points=120, T_max=5, type="class", X_test = X_heart[-heart_idx,], y_test =y_heart[-heart_idx])
print("heart hb / seq done")
res_A[[7]] <- hyperband(X=X_fire[fire_idx,], y=y_fire[fire_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_A, R=9000, type="reg", X_test = X_fire[-fire_idx,], y_test =y_fire[-fire_idx])
res_A[[8]] <- seqUD(X=X_fire[fire_idx,], y=y_fire[fire_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_A, n_points=120, T_max=5, type="reg", X_test = X_fire[-fire_idx,], y_test =y_fire[-fire_idx])
print("fire hb / seq done")
#n_iter is arbitrarily large we expect early stopping, 10% more time than grid search
res_A[[9]] <- bayes_search(X=X_heart[heart_idx,], y=y_heart[heart_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_A, n=100, type="class", X_test = X_heart[-heart_idx,], y_test =y_heart[-heart_idx], t_limit = 517)
res_A[[10]] <- bayes_search(X=X_fire[fire_idx,], y=y_fire[fire_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_A, n=100, type="reg", X_test = X_fire[-fire_idx,], y_test =y_fire[-fire_idx], t_limit = 550)
print("bayes A done")

res_A

res_A_df <- do.call(rbind, lapply(res_A, function(x) as.data.frame(x, row.names = NULL)))
write.csv(res_A_df, out_ffnn_A, row.names = FALSE)
res_A_df

## Medium/Large datasets - Group B
# 1920 combs

ffnn_params_B <- list(
  lr = c(0.001, 0.003, 0.01),           
  epochs = c(100, 200),                
  batch_size = c(64, 128),           
  weight_decay = c(0, 0.0005),           
  gamma = c(0.90, 0.99),                
  
  size1 = c(64, 128),
  act1 = c(1),
  size2 = c(32, 64),
  act2 = c(1),
  drop2 = c(0.0, 0.3),                 
  size3 = c(0, 32),                   
  act3 = c(1, 2),                     
  drop3 = c(0.0, 0.2)
)


# make grid and remove dups
grid_B <- expand.grid(ffnn_params_B, stringsAsFactors = FALSE)
grid_B$drop3[grid_B$size3 == 0] <- NA
grid_B$act3[grid_B$size3 == 0]  <- NA

grid_B <- unique(grid_B)
nrow(grid_B)

ffnn_dists_B <- list(
  lr = function(n) 10^runif(n, -3, -2),          # ~0.001–0.01
  epochs = function(n) sample(100:200, n, TRUE),
  batch_size = function(n) sample(c(64, 128, 256), n, TRUE),
  weight_decay = function(n) runif(n, 0, 0.001),
  gamma = function(n) runif(n, 0.90, 0.99),
  
  size1 = function(n) sample(64:128, n, TRUE),
  act1 = function(n) rep(1, n),
  size2 = function(n) sample(32:64, n, TRUE),
  act2 = function(n) rep(1, n),
  drop2 = function(n) runif(n, 0, 0.3),
  size3 = function(n) sample(0:16, n, TRUE),
  act3 = function(n) sample(c(1, 2, 3), n, TRUE),
  drop3 = function(n) runif(n, 0, 0.2)
)

ffnn_bounds_B <- list(
  lr = c(0.001, 0.01),
  epochs = c(100L, 200L),
  batch_size = c(64L, 256L),
  weight_decay = c(0, 0.001),
  gamma = c(0.90, 0.99),
  
  size1 = c(64L, 128L),
  act1 = c(1L, 2L),
  size2 = c(32L, 64L),
  act2 = c(1L, 2L),
  drop2 = c(0, 0.3),
  size3 = c(0L, 32L),
  act3 = c(1L, 3L),
  drop3 = c(0, 0.2)
)

out_ffnn_B <- "C:/Users/Karan/Desktop/HPO/results/ffnn_B.csv"

res_B <- list()

res_B[[1]] <- grid_search(X=X_bank[bank_idx,], y=y_bank[bank_idx], obj_func = obj_FFNN, gd = grid_B, type="class", X_test = X_bank[-bank_idx,], y_test =y_bank[-bank_idx])
res_B[[2]] <- random_search(X=X_bank[bank_idx,], y=y_bank[bank_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_B, n=1920, type="class", X_test = X_bank[-bank_idx,], y_test =y_bank[-bank_idx])

res_B[[3]] <- grid_search(X=X_bike[bike_idx,], y=y_bike[bike_idx], obj_func = obj_FFNN, gd = grid_B, type="reg", X_test = X_bike[-bike_idx,], y_test =y_bike[-bike_idx])
res_B[[4]] <- random_search(X=X_bike[bike_idx,], y=y_bike[bike_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_B, n=1920, type="reg", X_test = X_bike[-bike_idx,], y_test =y_bike[-bike_idx])

res_B[[5]] <- hyperband(X=X_bank[bank_idx,], y=y_bank[bank_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_B, R=9000, type="class", X_test = X_bank[-bank_idx,], y_test =y_bank[-bank_idx])
res_B[[6]] <- seqUD(X=X_bank[bank_idx,], y=y_bank[bank_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_B, n_points=100, T_max=3, type="class", X_test = X_bank[-bank_idx,], y_test =y_bank[-bank_idx])
print("hb / seq bank done")
res_B[[7]] <- hyperband(X=X_bike[bike_idx,], y=y_bike[bike_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_B, R=9000, type="reg", X_test = X_bike[-bike_idx,], y_test =y_bike[-bike_idx])
res_B[[8]] <- seqUD(X=X_bike[bike_idx,], y=y_bike[bike_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_B, n_points=100, T_max=3, type="reg", X_test = X_bike[-bike_idx,], y_test =y_bike[-bike_idx])
print("hb / seq bike done")
#n_iter is arbitrarily large we expect early stopping
res_B[[9]] <- bayes_search(X=X_bank[bank_idx,], y=y_bank[bank_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_B, n=200, type="class", X_test = X_bank[-bank_idx,], y_test =y_bank[-bank_idx], t_limit = 5918)
res_B[[10]] <- bayes_search(X=X_bike[bike_idx,], y=y_bike[bike_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_B, n=200, type="reg", X_test = X_bike[-bike_idx,], y_test =y_bike[-bike_idx], t_limit = 2585)
print("Bayes B done")


res_B_df <- do.call(rbind, lapply(res_B, function(x) as.data.frame(x, row.names = NULL)))
res_B_df <- rbind(res_B_df, read.csv("C:/Users/Karan/Desktop/HPO/results/ffnn_B.csv"))
write.csv(res_B_df, out_ffnn_B, row.names = FALSE)
res_B_df


## High Dimensional datasets - Group C
# 2048 combs
ffnn_params_C <- list(
  lr = c(0.001, 0.005),           
  epochs = c(100, 200),             
  batch_size = c(128, 256),            
  weight_decay = c(0, 0.0005),      
  gamma = c(0.90, 0.99),           
  
  size1 = c(128, 256),
  act1 = c(1),
  size2 = c(64, 128),
  act2 = c(1),
  drop2 = c(0.0, 0.3),               
  size3 = c(32, 64),
  act3 = c(1, 2),                      
  drop3 = c(0.0, 0.2)
)

# make grid and no dups here
grid_C <- expand.grid(ffnn_params_C, stringsAsFactors = FALSE)
nrow(grid_C)


ffnn_dists_C <- list(
  lr = function(n) 10^runif(n, -3, -2.3),        # ~0.001–0.005
  epochs = function(n) sample(100:200, n, TRUE),
  batch_size = function(n) sample(c(128, 256, 512), n, TRUE),
  weight_decay = function(n) runif(n, 0, 0.001),
  gamma = function(n) runif(n, 0.90, 0.99),
  
  size1 = function(n) sample(128:256, n, TRUE),
  act1 = function(n) rep(1, n),
  size2 = function(n) sample(64:128, n, TRUE),
  act2 = function(n) rep(1, n),
  drop2 = function(n) runif(n, 0, 0.3),
  size3 = function(n) sample(32:64, n, TRUE),
  act3 = function(n) sample(c(1, 2, 3), n, TRUE),
  drop3 = function(n) runif(n, 0, 0.2)
)

ffnn_bounds_C <- list(
  lr = c(0.001, 0.005),
  epochs = c(100L, 200L),
  batch_size = c(128L, 512L),
  weight_decay = c(0, 0.001),
  gamma = c(0.90, 0.99),
  
  size1 = c(128L, 256L),
  act1 = c(1L, 2L),
  size2 = c(64L, 128L),
  act2 = c(1L, 2L),
  drop2 = c(0, 0.3),
  size3 = c(32L, 64L),
  act3 = c(1L, 3L),
  drop3 = c(0, 0.2)
)

out_ffnn_C <- "C:/Users/Karan/Desktop/HPO/results/ffnn_C.csv"

res_C <- list()

res_C[[1]] <- grid_search(X=X_super[super_idx,], y=y_super[super_idx], obj_func = obj_FFNN, gd = grid_C, type="reg", X_test = X_super[-super_idx,], y_test =y_super[-super_idx])
res_C[[2]] <- random_search(X=X_super[super_idx,], y=y_super[super_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_C, n=2048, type="reg", X_test = X_super[-super_idx,], y_test =y_super[-super_idx])

res_C[[3]] <- hyperband(X=X_super[super_idx,], y=y_super[super_idx], obj_func = obj_FFNN, param_dists = ffnn_dists_C, R=9000, type="reg", X_test = X_super[-super_idx,], y_test =y_super[-super_idx])
res_C[[4]] <- seqUD(X=X_super[super_idx,], y=y_super[super_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_C, n_points=100, T_max=3, type="reg", X_test = X_super[-super_idx,], y_test =y_super[-super_idx])
print("hb / seq C done")
#n_iter is arbitrarily large we expect early stopping
res_C[[5]] <- bayes_search(X=X_super[super_idx,], y=y_super[super_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_C, n=200, type="reg", X_test = X_super[-super_idx,], y_test =y_super[-super_idx], t_limit = 2200)
print("FFNN done! YIPPERR!")


res_C_df <- do.call(rbind, lapply(res_C, function(x) as.data.frame(x, row.names = NULL)))
res_C_df <- rbind(res_C_df, read.csv("C:/Users/Karan/Desktop/HPO/results/ffnn_C.csv"))
write.csv(res_C_df, out_ffnn_C, row.names = FALSE)
res_C_df


#+++++++++++++++++++++++++++++++++++++++++++++++++#
########## Convolutional Neural Network ###########
#+++++++++++++++++++++++++++++++++++++++++++++++++#

## MNIST

mnist_params <- list(
  lr = c(0.001, 0.005),            # 2
  epochs = c(5, 50),               # 2
  batch_size = c(64, 128),         # 2
  weight_decay = c(0.0, 0.0005),   # 2
  gamma = c(0.90, 0.99),           # 2
  
  conv1 = c(32, 64),               # 2
  kernel1 = c(5),                  # 1
  pool1 = c(2),                    # 1
  
  conv2 = c(64, 128),              # 2
  kernel2 = c(3),                  # 1
  pool2 = c(2),                    # 1
  drop2 = c(0.3, 0.4),             # 2
  
  conv3 = c(128, 256),             # 2
  kernel3 = c(3),                  # 1
  pool3 = c(2),                    # 1
  drop3 = c(0.4, 0.5)              # 2
)

mnist_bounds <- list(
  lr = c(0.001, 0.005),
  epochs = c(5L, 50L),
  batch_size = c(64L, 128L),
  weight_decay = c(0, 0.0005),
  gamma = c(0.90, 0.99),
  
  conv1 = c(32L, 64L),
  kernel1 = c(3L, 7L),
  pool1 = c(2L, 3L),
  
  conv2 = c(64L, 128L),
  kernel2 = c(3L, 7L),
  pool2 = c(2L, 3L),
  drop2 = c(0.3, 0.4),
  
  conv3 = c(128L, 256L),
  kernel3 = c(3L, 5L),
  pool3 = c(2L, 3L),
  drop3 = c(0.4, 0.5)
)

mnist_dists <- list(
  lr = function(n) 10^runif(n, -3, -2.3),
  epochs = function(n) sample(5:50, n, TRUE),
  batch_size = function(n) sample(c(64,128), n, TRUE),
  weight_decay = function(n) runif(n, 0, 0.0005),
  gamma = function(n) runif(n, 0.90, 0.99),
  
  conv1 = function(n) sample(c(32,64), n, TRUE),
  kernel1 = function(n) sample(c(3,5,7), n, TRUE),
  pool1 = function(n) rep(2, n),
  
  conv2 = function(n) sample(c(64,128), n, TRUE),
  kernel2 = function(n) sample(c(3,5), n, TRUE),
  pool2 = function(n) rep(2, n),
  drop2 = function(n) runif(n, 0.3, 0.4),
  
  conv3 = function(n) sample(c(128,256), n, TRUE),
  kernel3 = function(n) rep(3, n),
  pool3 = function(n) rep(2, n),
  drop3 = function(n) runif(n, 0.4, 0.5)
)

# run it #

out_cnn <- "C:/Users/Karan/Desktop/HPO/results/mnist.csv"
res_CNN <- list()

res_CNN[[1]] <- grid_search(setname="MNIST", obj_func = obj_CNN, param_values = mnist_params)
res_CNN[[2]] <- random_search(setname="MNIST", obj_func = obj_CNN, param_dists = mnist_dists, n=1024)
res_CNN[[3]] <- bayes_search(setname="MNIST", obj_func = obj_CNN, param_bounds = mnist_bounds, n=250, t_limit = 57600)
res_CNN[[4]] <- hyperband(setname="MNIST", obj_func = obj_CNN, param_dists = mnist_dists, R = 6000)
res_CNN[[5]] <- seqUD(setname="MNIST", obj_func = obj_CNN, param_bounds = mnist_bounds, n_points=100, T_max=3)

res_CNN_df <- do.call(rbind, lapply(res_CNN, function(x) as.data.frame(x, row.names = NULL)))
res_CNN_df <- rbind(res_CNN_df, read.csv(out_cnn))
write.csv(res_CNN_df, out_cnn, row.names = FALSE)
res_CNN_df


## CNN test

mnist_params <- list(
  lr = 0.005,
  epochs = 5, 
  batch_size = 128,
  weight_decay = 0.0005,
  gamma = 0.95,
  
  conv1 = 32,  kernel1 = 5,  pool1 = 2,
  conv2 = 64,  kernel2 = 4,  pool2 = 2,
  conv3 = 128, kernel3 = 3,  pool3 = 2,
  
  drop2 = 0.4,
  drop3 = 0.5
)

# CIFAR-10
cifar10_params <- list(
  lr = 0.01,
  epochs = 10,
  batch_size = 128,
  weight_decay = 0.0001,
  gamma = 0.9,
  
  conv1 = 64,   kernel1 = 3, pool1 = 2,
  conv2 = 128,  kernel2 = 3, pool2 = 2,
  conv3 = 256,  kernel3 = 3, pool3 = 2,
  conv4 = 512,  kernel4 = 3, pool4 = 2,
  
  drop2 = 0.4,
  drop3 = 0.5,
  drop4 = 0.5
)

# CIFAR-100
cifar100_params <- list(
  lr = 0.01,
  epochs = 15,
  batch_size = 128,
  weight_decay = 0.0001,
  gamma = 0.9,
  
  conv1 = 64,   kernel1 = 3, pool1 = 2,
  conv2 = 128,  kernel2 = 3, pool2 = 2,
  conv3 = 256,  kernel3 = 3, pool3 = 2,
  conv4 = 512,  kernel4 = 3, pool4 = 2,
  conv5 = 512,  kernel5 = 3, pool5 = 2,
  
  drop2 = 0.3,
  drop3 = 0.4,
  drop4 = 0.5,
  drop5 = 0.5
)

# Benchmark runs

obj_CNN("MNIST", mnist_params, mode="test")

system.time( obj_CNN("MNIST", mnist_params) )
system.time( obj_CNN("CIFAR10", cifar10_params) )
system.time( obj_CNN("CIFAR100", cifar100_params) )




mnist_big <- list(
  lr = 0.005,
  epochs = 50, 
  batch_size = 128,
  weight_decay = 0.0005,
  gamma = 0.95,
  
  conv1 = 64,   kernel1 = 5, pool1 = 2,
  conv2 = 128,  kernel2 = 5, pool2 = 2, drop2 = 0.4,
  conv3 = 256,  kernel3 = 3, pool3 = 2, drop3 = 0.5
)

# CIFAR-10: deeper & wider (roughly VGG11-ish scale)
cifar10_big <- list(
  lr = 0.01,
  epochs = 100,
  batch_size = 128,
  weight_decay = 0.0001,
  gamma = 0.9,
  
  conv1 = 128,  kernel1 = 3, pool1 = 2,
  conv2 = 256,  kernel2 = 3, pool2 = 2, drop2 = 0.4,
  conv3 = 512,  kernel3 = 3, pool3 = 2, drop3 = 0.5,
  conv4 = 512,  kernel4 = 3, pool4 = 2, drop4 = 0.5
)

# CIFAR-100: heaviest — deeper + doubled width (close to VGG16 scale)
cifar100_big <- list(
  lr = 0.01,
  epochs = 200,
  batch_size = 128,
  weight_decay = 0.0001,
  gamma = 0.9,
  
  conv1 = 128,  kernel1 = 3, pool1 = 2,
  conv2 = 256,  kernel2 = 3, pool2 = 2, drop2 = 0.3,
  conv3 = 512,  kernel3 = 3, pool3 = 2, drop3 = 0.4,
  conv4 = 512,  kernel4 = 3, pool4 = 2, drop4 = 0.5,
  conv5 = 1024, kernel5 = 3, pool5 = 2, drop5 = 0.5
)

# Benchmark runs
system.time( obj_CNN("MNIST", mnist_big) )
system.time( obj_CNN("CIFAR10", cifar10_big) )
system.time( obj_CNN("CIFAR100", cifar100_big) )


######################################################
############ Re runs ###############################
####################################################

re_run_res <- list()

re_run_res[[1]] <- bayes_search(X=X_heart[heart_idx,], y=y_heart[heart_idx], obj_func = obj_RF, param_bounds = rf_bounds, n=2, type="class", X_test = X_heart[-heart_idx,], y_test =y_heart[-heart_idx])
re_run_res[[2]] <- bayes_search(X=X_fire[fire_idx,], y=y_fire[fire_idx], obj_func = obj_RF, param_bounds = rf_bounds, n=3, type="reg", X_test = X_fire[-fire_idx,], y_test =y_fire[-fire_idx])
re_run_res[[3]] <- seqUD(X=X_heart[heart_idx,], y=y_heart[heart_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_A, n_points=100, T_max=2, type="class", X_test = X_heart[-heart_idx,], y_test =y_heart[-heart_idx])
re_run_res[[4]] <- seqUD(X=X_fire[fire_idx,], y=y_fire[fire_idx], obj_func = obj_FFNN, param_bounds = ffnn_bounds_A, n_points=100, T_max=2, type="reg", X_test = X_fire[-fire_idx,], y_test =y_fire[-fire_idx])
re_run_res[[5]] <- bayes_search(X=X_heart[heart_idx,], y=y_heart[heart_idx], obj_func = obj_XGB, param_bounds = xgb_bounds, n=100, type="class", X_test = X_heart[-heart_idx,], y_test =y_heart[-heart_idx], t_limit = 20)
re_run_res[[6]] <- bayes_search(X=X_fire[fire_idx,], y=y_fire[fire_idx], obj_func = obj_XGB, param_bounds = xgb_bounds, n=100, type="reg", X_test = X_fire[-fire_idx,], y_test =y_fire[-fire_idx], t_limit = 20)

re_run_res_df <- do.call(rbind, lapply(re_run_res, function(x) as.data.frame(x[1:8], row.names = NULL)))
write.csv(re_run_res_df, "C:/Users/Karan/Desktop/HPO/results/re_run.csv", row.names = FALSE)
re_run_res_df

