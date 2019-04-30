library(h2o)

c1=h2o.init(max_mem_size = "4G", 
            nthreads = 2, 
            ip = "localhost", 
            port = 54321)

h2o.removeAll()

# read csv file housing.csv
housing = read.csv("c:/housing.csv")

# add dummy variables for categorical column 'ocean_proximity'
library(dummies)
d1 = dummy(housing$ocean_proximity)
housing = housing[,-10]
housing = cbind(d1, housing[,])

#shorten column names
colnames(housing)= c("ocean_1h","inland","island","near_bay","near_ocean",colnames(housing[,6:14]))

#impute missing values with median
housing$total_bedrooms[is.na(housing$total_bedrooms)] = median(housing$total_bedrooms, na.rm = TRUE)

#library(Hmisc)
#housing$total_bedrooms = impute(housing$total_bedrooms, "median")

#replace mean values with total ones
housing$mean_bedrooms = housing$total_bedrooms/housing$households
housing$mean_rooms = housing$total_rooms/housing$households
housing = housing[,-(9:10)]


# calculate correlation matrix
#mycorr = cor(housing)
#mycorr

# Eigensystem Analysis
#eigen(cor(housing))$values

# Condition Number: Ratio of max to min Eigen values of the correlation matrix
#max(eigen(cor(housing))$values)/min(eigen(cor(housing))$values)
#kappa(cor(housing), exact = TRUE)

# PCA steps
#housingx = housing[,-12]
#housingx_pca = prcomp(housingx, center = TRUE, scale = TRUE)
#plot(housingx_pca, type = "l")
#summary(housingx_pca)

#  here is the rotation matrix :
#print(housingx_pca$rotation)


# the new rotated components are contained in housing.pca$x
# double check that the new principal components are orthogonal
#cor(housingx_pca$x)

# We can also look at the correlation between the original predictor variables
# and the new principal components
#cor(cbind(housing[,-12],data.frame(housingx_pca$x)))

# Let's make a data frame with Housing in the first column, and
# the principal components in the rest of the columns
housing = cbind(housing[,12],housing[,-12])
colnames(housing)[1] <- "median_house_value"

# now, let's look at the correlation matrix, showing the correlation of
# median_house_price to each principal component
#cor(housing_pca)[,1]

allrows <- 1:nrow(housing)

set.seed(1981)

trainrows <- sample(allrows, replace = F, size = 0.9*length(allrows))
test_cvrows <- allrows[-trainrows]
testrows <- sample(test_cvrows, replace=F, size = 0.5*length(test_cvrows))
cvrows <- test_cvrows[-which(test_cvrows %in% testrows)]

train <- as.h2o(housing[trainrows,])
test <- as.h2o(housing[testrows,])
valid <- as.h2o(housing[cvrows,])

nrow(train) + nrow(test) + nrow(valid) == nrow(housing)

#######################################################################

#set parameter space
activation_opt <- c("Rectifier","RectifierWithDropout", "Maxout","MaxoutWithDropout")
hidden_opt <- list(10,20,30,40)
l1_opt <- c(0,1e-3,1e-5)
l2_opt <- c(0,1e-3,1e-5)

hyper_params <- list( activation=activation_opt,
                      hidden=hidden_opt,
                      l1=l1_opt,
                      l2=l2_opt )

#set search criteria
search_criteria <- list(strategy = "RandomDiscrete", max_models=10)

#train model
dl_grid <- h2o.grid("deeplearning"
                    ,grid_id = "deep_learn"
                    ,hyper_params = hyper_params
                    ,search_criteria = search_criteria
                    ,training_frame = train
                    ,x= 2:14
                    ,y= 1
                    ,nfolds = 10
                    ,epochs = 1000)

#get best model
d_grid <- h2o.getGrid("deep_learn",sort_by = "rmse")
best_dl_model <- h2o.getModel(d_grid@model_ids[[1]])
h2o.performance (best_dl_model,xval = T)
#CV Accuracy - 84.7%

#######################################################################

nn_model2 = h2o.deeplearning(x = 2:14, y = 1 , training_frame = train, validation_frame = valid,
                            hidden = 30, epoch = 1000, stopping_metric = "RMSE", verbose = TRUE)


pred_nn2 = h2o.predict(nn_model2, newdata = test)
cor(pred_nn2, test[,12])

#######################################################################

nn_model3 = h2o.deeplearning(x = 2:14, y = 1 , training_frame = train, validation_frame = valid,
                             hidden = 20, epoch = 10000, stopping_metric = "RMSE", verbose = TRUE)


pred_nn3 = h2o.predict(nn_model3, newdata = test)
cor(pred_nn3, test[,12])