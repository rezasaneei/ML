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

## PCA steps
##housingx = housing[,-12]
##housingx_pca = prcomp(housingx, center = TRUE, scale = TRUE)
##plot(housingx_pca, type = "l")
##summary(housingx_pca)

# Let's make a data frame with Housing in the first column, and
# the principal components in the rest of the columns
##housing_pca = cbind(housing[,12],data.frame(housingx_pca$x))
##colnames(housing_pca)[1] <- "median_house_value"

# Create Vector of Column Max and Min Values
maxs <- apply(housing[,], 2, max)
mins <- apply(housing[,], 2, min)

# Use scale() and convert the resulting matrix to a data frame
housing_scaled <- as.data.frame(scale(housing[,],center = mins, scale = maxs - mins))

# Check out results
print(head(housing_scaled,4))

set.seed(1981)

# sample data for train or test
sample = sample.int(n = nrow(housing_scaled), size = floor(.8*nrow(housing_scaled)), replace = TRUE)
train = as.data.frame(housing_scaled[sample, ])
test  = as.data.frame(housing_scaled[-sample, ])

n <- names(train)
f <- as.formula(paste("median_house_value ~", paste(n[!n %in% "median_house_value"], collapse = " + ")))

library(nnet)
library(caret)
library(doParallel)

cl1 = makeCluster(7)
registerDoParallel(cl1)
tunegrid = expand.grid(size = c(20,25,30), decay = c(0.01,0.001,0.0001))
fitControl = trainControl(method="cv", number=10, allowParallel = TRUE)
t1 = Sys.time()
nnet_model = train(f, data = train, method = "nnet", maxit = 10000, trControl = fitControl, tuneGrid = tunegrid, metric = "RMSE")
t2 = Sys.time()
predY = predict(nnet_model, test[,-12])
cor(predY, test[12])
t2-t1
print(nnet_model)
stopCluster(cl1)

cl1 = makeCluster(7)
registerDoParallel(cl1)
tunegrid2 = expand.grid(size = c(30,40), decay = c(1e-04,1e-05))
fitControl2 = trainControl(method="cv", number=10, allowParallel = TRUE)
t1 = Sys.time()
nnet_model2 = train(f, data = train, method = "nnet", maxit = 10000, abstol = 1.0e-6 , reltol = 1.0e-10, trControl = fitControl2, tuneGrid = tunegrid2, metric = "RMSE")
t2 = Sys.time()
predY2 = predict(nnet_model2, test[,-12])
cor(predY2, test[12])
t2-t1
print(nnet_model2)
stopCluster(cl1)


t1 = Sys.time()
nnet_model3 = nnet(f, data = train, method = "nnet", maxit = 50000,  wts = nnet_model2$finalModel$wts ,abstol = 1.0e-7 , reltol = 1.0e-10, size = 40, decay = 1.0e-6)
t2 = Sys.time()
predY3 = predict(nnet_model3, test[,-12])
cor(predY2, test[12])
t2-t1
print(nnet_model3)

t1 = Sys.time()
nnet_model4 = nnet(f, data = train, method = "nnet", maxit = 50000,  Wts = nnet_model3$wts ,abstol = 1.0e-8 , reltol = 1.0e-11, size = 40, decay = 1.0e-8)
t2 = Sys.time()
predY4= predict(nnet_model4, test[,-12])
cor(predY4, test[12])
t2-t1
print(nnet_model4)

optim_model1 = optim(nnet_model3   )