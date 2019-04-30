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
#maxs <- apply(housing[,], 2, max)
#mins <- apply(housing[,], 2, min)

# Use scale() and convert the resulting matrix to a data frame
#housing_scaled <- as.data.frame(scale(housing[,],center = mins, scale = maxs - mins))

# Check out results
#print(head(housing_scaled,4))

set.seed(1981)

# sample data for train or test
sample = sample.int(n = nrow(housing), size = floor(.8*nrow(housing)), replace = FALSE)
train = housing[sample, ] 
test  = housing[-sample, ]

n <- names(train)
f <- as.formula(paste("median_house_value ~", paste(n[!n %in% "median_house_value"], collapse = " + ")))


###############################################################
library(caret)
library(doParallel)
cl1 = makeCluster(6)
registerDoParallel(cl1)

ctrl <- trainControl(method="repeatedcv",   # 10-fold cross validation
                     repeats=3,		    # do 5 repititions of cv
                     number = 10,
                     allowParallel = TRUE)

t1 = Sys.time()

#Train and Tune the SVM
svm_tune <- train(f, data = train,
                  method = "svmLinear",   # Radial kernel
                  tuneLength = 10,				# 10 values of the cost function
                  preProc = c("center","scale"),  # Center and scale data
                  metric="RMSE",
                  trControl=ctrl)
t2 = Sys.time()
t2-t1

print(svm_tune)
summary(svm_tune)
plot(svm_tune)
predY1 <- predict(svm_tune,test[,-12])
cor(predY1, test[,12])
stopCluster(cl1)
#############################################################

cl1 = makeCluster(6)
registerDoParallel(cl1)

ctrl <- trainControl(method="repeatedcv",   # 10-fold cross validation
                     repeats=3,		    # do 5 repititions of cv
                     number = 10,
                     allowParallel = TRUE)

t1 = Sys.time()

#Train and Tune the SVM
svm_tune2 <- train(f, data = train,
                  method = "svmRadial",   # Radial kernel
                  tuneLength = 10,				# 10 values of the cost function
                  preProc = c("center","scale"),  # Center and scale data
                  metric="RMSE",
                  trControl=ctrl)
t2 = Sys.time()
t2-t1
print(svm_tune2)
summary(svm_tune2)
plot(svm_tune2)
predY2 <- predict(svm_tune2,test[,-12])
cor(predY2, test[,12])
stopCluster(cl1)
