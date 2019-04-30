exam = read.csv("c://exam.csv")
summary(exam)


exam$smoking_[exam$smoking=="little"] = 1
exam$smoking_[exam$smoking=="normal"] = 2
exam$smoking_[exam$smoking=="much"] = 3
#############################
exam$radeon_[exam$radeon=="little"] = 1
exam$radeon_[exam$radeon=="normal"] = 2
exam$radeon_[exam$radeon=="much"] = 3
#############################
exam$exercise_[exam$exercise=="little"] = 3
exam$exercise_[exam$exercise=="normal"] = 2
exam$exercise_[exam$exercise=="much"] = 1
#############################
exam$risk[exam$cancer=="low"] = 0
exam$risk[exam$cancer=="high"] = 1

print(exam_scaled)

exam = exam[,-(1:4)]

maxs <- apply(exam[,], 2, max)
mins <- apply(exam[,], 2, min)

# Use scale() and convert the resulting matrix to a data frame
exam_scaled <- as.data.frame(scale(exam[,],center = mins, scale = maxs - mins))

library(neuralnet)
n <- names(exam_scaled)
f <- as.formula(paste("risk ~", paste(n[!n %in% "risk"], collapse = " + ")))

##########################################################################################################

system.time(nn1 <- neuralnet(f,data=exam_scaled, lifesign = "full", stepmax = 100000))
plot(nn1)

