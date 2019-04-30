# PCA built-in train function

# read csv file housing.csv
housing = read.csv("housing.csv")

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
mycorr = cor(housing)
mycorr

# Eigensystem Analysis
eigen(cor(housing))$values

# Condition Number: Ratio of max to min Eigen values of the correlation matrix
max(eigen(cor(housing))$values)/min(eigen(cor(housing))$values)
kappa(cor(housing), exact = TRUE)

housing = cbind(housing[,12],housing[,-12])
colnames(housing)[1] = "median_house_value"

set.seed(1981)

# sample data for train or test
sample = sample.int(n = nrow(housing), size = floor(.8*nrow(housing)), replace = FALSE)
train = housing[sample, ] 
test  = housing[-sample, ]

head(train)

nrow(train) + nrow(test) == nrow(housing)

# simple random forest

library(randomForest)
library(caret)
library(doParallel)
cl1 = makeCluster(4)
registerDoParallel(cl1)

tunegrid <- expand.grid(mtry=c(5:9))
fitControl <- trainControl(preProcOptions = c("BoxCox", "center", "scale", "pca"), method="cv", number=10, allowParallel = TRUE)
rf_model = train(train[,-1], train[,1], method = "rf", trControl = fitControl, tuneGrid = tunegrid, metric = "RMSE")
print(rf_model)
plot(rf_model)
predY = predict(rf_model , test[,-1])
cor(predY, test[,1])
test_rmse = sqrt(mean((predY - test[,1])^2))
test_rmse
