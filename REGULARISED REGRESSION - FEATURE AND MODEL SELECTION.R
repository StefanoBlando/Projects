
##DATA PROCESSING

library(glmnet)
library(MASS)
library(SIS)
library(caret)
library(ggplot2)
library(car)
library(dplyr)

# Loading data
data <- read.csv("C:/Users/stepb/Desktop/BIG DATA ANALYSIS FOR ECONOMICS AND FINANCE/ASSIGNMENT 2/OnlineNewsPopularity.csv")

# data structure and cleaning
dim(data)
str(data)
glimpse(data)

data_clean <- data[data$shares <= 1e5, ]

#setting seed to reproduce the analysis
set.seed(123)

# Random selection: 80% of the data for estimation, 20% for testing
est_indices <- sample(1:nrow(data_clean), 0.8 * nrow(data_clean))
train_data <- data_clean[est_indices, ]
test_data <- data_clean[-est_indices, ]


# Defining predictors and outcome
predictors <- colnames(train_data[,c(3, 4, 8, 10:19, 39, 45:48)])
outcome <- colnames(train_data[c(61)])

#Defining X e Y for estimation and test sets
X_train <- train_data[, predictors]
y_train <- train_data[, outcome]
X_test  <- test_data[, predictors]
y_test  <- test_data[, outcome] 


## FULL LINEAR MODEL 

full_model <- lm(y_train~ ., data = X_train)
summary(full_model)
plot(full_model)

predictions_full_model <- predict(full_model, newdata = X_test)
MSE_full_model <- sqrt(1/nrow(X_test)*sum((y_test-predictions_full_model)^2))
summary(predictions_full_model)


## FORWARD STEPWISE SELECTION 

#model framework
min_model <- lm(y_train ~ 1, data = X_train)
max_model <-lm(y_train~ ., data= X_train)

# step forward model
forward_model_AIC <- step(min_model, direction='forward', scope=formula(max_model), trace=0)
forward_model_AIC$anova
forward_model_AIC$coefficients
summary(forward_model_AIC)
#predictions
predictions_forward_model_AIC <- predict(forward_model_AIC, newdata= X_test,type="response")
MSE_forward_model_AIC <- sqrt(1/nrow(X_test)*sum((y_test-predictions_forward_model_AIC)^2))
summary(predictions_forward_model_AIC)

# stepBIC forward model
n=nrow(X_train)
forward_model_BIC <- stepAIC(min_model, direction='forward', scope=formula(max_model),k=log(n), trace=0)
forward_model_BIC$anova
forward_model_BIC$coefficients
summary(forward_model_BIC)

#predictions
predictions_forward_model_BIC <- predict(forward_model_BIC, newdata = X_test, type="response")
MSE_forward_model_BIC <- sqrt(1/nrow(X_test)*sum((y_test-predictions_forward_model_BIC)^2))
summary(predictions_forward_model_BIC)

## RIDGE REGRESSION

# input processing
y = y_train
x = as.matrix(X_train)

# base model
ridge_model = glmnet(x=x, y=y, alpha = 0, family = "poisson", nlambda = 250)
plot(ridge_model$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(ridge_model$beta)
image(ridge_model$beta)
matplot(t(ridge_model$beta), type = "l")

#cross-validation
set.seed(123)
cv_ridge = cv.glmnet(x=x, y=y, alpha=0, family="poisson", nlambda=250)
plot(log(cv_ridge$lambda), cv_ridge$cvm,type = "b", pch=19)
plot(cv_ridge)

mridge = which.min(cv_ridge$cvm) 

plot(log(cv_ridge$lambda)[(mridge-10):(mridge+10)], cv_ridge$cvm[(mridge-10):(mridge+10)],type = "b", pch=19)
abline(v=log(cv_ridge$lambda[mridge]), lty=4)  

# best selected model
ridge_model_cv = glmnet(x=x, y=y, family = "poisson", alpha=0,lambda = cv_ridge$lambda[mridge])

round(cbind(rbind(ridge_model_cv$a0, ridge_model_cv$beta)[-which(is.na(coef(full_model))),],coef(full_model)[-which(is.na(coef(full_model)))]),5)
predictions_ridge = predict(ridge_model_cv, newx = as.matrix(X_test), type="response")

# predictions and visualization
plot(y_test, predictions_ridge, pch=19)
abline(b=1, a=0, col=2, lwd=2)

ggplot(X_test, aes(x=predictions_ridge, y=y_test)) + 
  geom_point() +
  geom_abline(intercept=0, slope=1) +
  labs(x='Predicted Values', y='Actual Values', title='Predicted vs. Actual Values')

#MSE
MSE_ridge <- sqrt(1/nrow(X_test)*sum((y_test-predictions_ridge)^2))


## LASSO REGRESSION

# base model
lasso_model <- glmnet(x=x, y=y, alpha = 1, family = "poisson", nlambda = 250)
plot(lasso_model$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(lasso_model$beta)
image(lasso_model$beta)
matplot(t(lasso_model$beta), type = "l")

# cross validation
set.seed(123)
cv_lasso = cv.glmnet(x=x, y=y, family="poisson", nlambda=250, alpha=1)
plot(log(cv_lasso$lambda), cv_lasso$cvm,type = "b", pch=19)
plot(cv_lasso)

mlasso = which.min(cv_lasso$cvm)

plot(log(cv_lasso$lambda)[(mlasso-10):(mlasso+10)], cv_lasso$cvm[(mlasso-10):(mlasso+10)],type = "b", pch=19)
abline(v=log(cv_lasso$lambda[mlasso]), lty=4) 

# best selected model 
lasso_model_cv = glmnet(x=x, y=y, family = "poisson", alpha=1,lambda = cv_lasso$lambda[mlasso])
round(cbind(rbind(lasso_model_cv$a0, lasso_model_cv$beta),rbind(ridge_model_cv$a0, ridge_model_cv$beta)),5)

# predictions and visualization
predictions_lasso = predict(lasso_model_cv, newx = as.matrix(X_test),type="response")
plot(y_test, predictions_lasso, pch=19)
abline(b=1, a=0, col=2, lwd=2)

#MSE
MSE_lasso <-sqrt(1/nrow(X_test)*sum((y_test-predictions_lasso)^2))

## ELASTIC NET REGRESSION
# base model
elasticnet_model = glmnet(x=x, y=y, family = "poisson", alpha=0.5)
plot(elasticnet_model$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(elasticnet_model$beta)
image(elasticnet_model$beta)
matplot(t(elasticnet_model$beta), type = "l")

lgrid = elasticnet_model$lambda
agrid = seq(0,1,length.out=25)

# cross validation
set.seed(123)
cvloop = cv.glmnet(x=x, y=y, family="poisson", alpha=agrid[1], lambda = lgrid)
cvloop = cbind(rep(agrid[1],length(cvloop$lambda)),cvloop$lambda,cvloop$cvm)
for(i in 2:length(agrid)){
  res_i =   cv.glmnet(x=x, y=y, family="poisson", alpha=agrid[i], lambda = lgrid)
  cvloop = rbind(cvloop, cbind(rep(agrid[i],length(res_i$lambda)),res_i$lambda,res_i$cvm))
}
cvloop = as.data.frame(cvloop)
names(cvloop) <- c("alpha","lambda","cvm")

ggplot(cvloop, aes(x=lambda, y=cvm, group=alpha, color=alpha))+geom_line()

alphastar = cvloop[which.min(cvloop$cvm),1]
lambdastar = cvloop[which.min(cvloop$cvm),2]

# best selected model
elasticnet_model_cv = glmnet(x=x, y=y, family = "poisson", alpha = alphastar, lambda = lambdastar)

round(cbind(rbind(lasso_model_cv$a0, lasso_model_cv$beta),rbind(ridge_model_cv$a0, ridge_model_cv$beta),rbind(elasticnet_model_cv$a0,elasticnet_model_cv$beta)),5)

# predictions and visualization
predictions_elasticnet = predict(elasticnet_model_cv, newx = as.matrix(X_test))
plot(y_test, predictions_elasticnet, pch=19)
abline(b=1, a=0, col=2, lwd=2)

#MSE
MSE_elasticnet <-sqrt(1/nrow(X_test)*sum((y_test-predictions_elasticnet)^2))



## CONCLUSIONS

MSE_values <- c(MSE_full_model, MSE_forward_model_AIC, MSE_forward_model_BIC, MSE_ridge, MSE_lasso, MSE_elasticnet)
MSE_names <- c("MSE_full_model", "MSE_forward_model_AIC", "MSE_forward_model_BIC", "MSE_ridge", "MSE_lasso", "MSE_elasticnet")

min_index <- which.min(MSE_values)
min_name <- MSE_names[min_index]
min_value <- MSE_values[min_index]

cat("Minimum MSE:", min_name, "(", min_value, ")\n")


The best model is LASSO.

The effect of global sentiment polarity seems to be influencing the popularity of online news, but said effect 
has to be calibrated by the rate of positive or negative words of the article, with a correlation 
that tends to prefer neative words. 