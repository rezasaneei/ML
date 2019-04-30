# PCA values nonly

# read csv file housing.csv
housing = read.csv(file = "./regression.csv", header = FALSE)

# add dummy variables for categorical column 'ocean_proximity'
# library(dummies)
# d1 = dummy(housing$ocean_proximity)
# housing = housing[,-10]
# housing = cbind(d1, housing[,])

#shorten column names
# colnames(housing)= c("ocean_1h","inland","island","near_bay","near_ocean",colnames(housing[,6:14]))

#impute missing values with median
# housing$total_bedrooms[is.na(housing$total_bedrooms)] = median(housing$total_bedrooms, na.rm = TRUE)

#library(Hmisc)
#housing$total_bedrooms = impute(housing$total_bedrooms, "median")

#replace mean values with total ones
# housing$mean_bedrooms = housing$total_bedrooms/housing$households
# housing$mean_rooms = housing$total_rooms/housing$households
# housing = housing[,-(9:10)]

housing$idx = seq.int(nrow(housing))
housing$num = 1

housing = housing[,c(13, 14, 1:12)]

write.csv(housing, file = "regression_data.csv", col.names = FALSE)
