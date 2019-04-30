# PCA values nonly

# read csv file housing.csv
housing = read.csv("./R/housing.csv")

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


# housing$idx = seq.int(nrow(housing))
# housing$num = seq.int(nrow(housing))
# 
# housing = housing[,c(15, 16, 1:14)]
# housing = housing[,c(1:12,14:16,13)]
# 
# 
# sample = sample.int(n = nrow(housing), size = floor(.8*nrow(housing)), replace = FALSE)
# train = housing[sample, ] 
# test  = housing[-sample, ]
# 
# 
# write.csv(train[,-16], file = "housing_train_features.csv")
# write.csv(train[,16], file = "housing_train_response.csv")
# write.csv(test[,-16], file = "housing_test_features.csv")
# write.csv(test[,16], file = "housing_test_response.csv")



# calculate correlation matrix
mycorr = cor(housing)
mycorr

# Eigensystem Analysis
eigen(cor(housing))$values

# Condition Number: Ratio of max to min Eigen values of the correlation matrix
max(eigen(cor(housing))$values)/min(eigen(cor(housing))$values)
kappa(cor(housing), exact = TRUE)

# PCA steps
housingx = housing[,-12]
housingx_pca = prcomp(housingx, center = TRUE, scale = TRUE)
plot(housingx_pca, type = "l")
summary(housingx_pca)

#  here is the rotation matrix :
print(housingx_pca$rotation)


# the new rotated components are contained in housing.pca$x
# double check that the new principal components are orthogonal
cor(housingx_pca$x)

# We can also look at the correlation between the original predictor variables
# and the new principal components
cor(cbind(housing[,-12],data.frame(housingx_pca$x)))

# Let's make a data frame with Housing in the first column, and
# the principal components in the rest of the columns
housing_pca = cbind(housing[,12],data.frame(housingx_pca$x))
colnames(housing_pca)[1] <- "median_house_value"

# now, let's look at the correlation matrix, showing the correlation of
# median_house_price to each principal component
cor(housing_pca)[,1]

set.seed(1981)

# sample data for train or test
sample = sample.int(n = nrow(housing_pca), size = floor(.8*nrow(housing_pca)), replace = FALSE)
train = housing_pca[sample, ] 
test  = housing_pca[-sample, ]

head(train)

nrow(train) + nrow(test) == nrow(housing_pca)

# simple random forest

library(randomForest)
library(caret)
library(doParallel)
cl1 = makeCluster(5)
registerDoParallel(cl1)

tunegrid <- expand.grid(mtry = 6)
fitControl <- trainControl(method="cv", number=10, allowParallel = TRUE)
t1 = Sys.time()
rf_model = train(x = train[,-1], y = train[,1] , method = "rf", trControl = fitControl, tuneGrid = tunegrid, metric = "RMSE" )
t2 = Sys.time()
print(rf_model)
plot(rf_model)
predY = predict(rf_model , test[,-1])
cor(predY, test[,1])
test_rmse = sqrt(mean((predY - test[,1])^2))
test_rmse
t2-t1
stopCluster(cl1)