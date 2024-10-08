#Libraries 
library(glmnet)
library(pROC)
library(class)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(PRROC)
library(knitr)
library(kableExtra)
library(neuralnet)
library(h2o)

#loading data
data <- read.csv("C:/Users/stepb/Desktop/BIG DATA ANALYSIS FOR ECONOMICS AND FINANCE/ASSIGNMENT 2/OnlineNewsPopularity.csv")

# Select candidate predictors
predictors <- data[, c(3, 4, 8, 10:19, 39, 45:48)]

# Define the target variable
target <- ifelse(data$shares > 1000, 1, 0)

# Split the data into training, validation, and testing sets
set.seed(123)
train.index <- sample(1:nrow(data), 0.7 * nrow(data))
valid.index <- sample(setdiff(1:nrow(data), train.index), 0.2 * nrow(data))
test.index <- setdiff(1:nrow(data), union(train.index, valid.index))

train.data <- predictors[train.index, ]
train.target <- target[train.index]
valid.data <- predictors[valid.index, ]
valid.target <- target[valid.index]
test.data <- predictors[test.index, ]
test.target <- target[test.index]


#####################################
##### LOGISTIC REGRESSION ###########
#####################################
#base model
logistic.model <- glm(train.target ~ ., data = train.data , family="binomial")

#data processing
dmx.train <- as.matrix(train.data)
dmx.valid <- as.matrix(valid.data)
dmx.test <- as.matrix(test.data)

dmy.train <- as.matrix(train.target)
dmy.valid <- as.matrix(valid.data)
dmy.test <- as.matrix(test.target)

#ridge regression
out.ridge = glmnet(x=train.data, y=train.target, alpha = 0, family = "binomial", nlambda = 250)
plot(out.ridge$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(out.ridge$beta)
image(out.ridge$beta)
matplot(t(out.ridge$beta), type = "l")

#select lambda and find best cv ridge
set.seed(123)
cv.logistic.ridge.model <- cv.glmnet(dmx.train, dmy.train, family = "binomial", nlambda=250, alpha=0, standardize=TRUE)
ridge = glmnet(x=train.data, y=train.target, family="binomial", alpha=0, lambda = cv.logistic.ridge.model$lambda.min)

# lasso 
out.lasso = glmnet(x=train.data, y=train.target, alpha = 1, family = "binomial", nlambda = 250)
plot(out.lasso$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(out.lasso$beta)
image(out.lasso$beta)
matplot(t(out.lasso$beta), type = "l")

#select lambda and find best cv lasso
set.seed(789)
cv.logistic.lasso.model <- cv.glmnet(dmx.train, dmy.train, family = "binomial", alpha = 1,standardize=TRUE)
lasso = glmnet(x=train.data, y=train.target, family="binomial", alpha=1, lambda = cv.logistic.lasso.model$lambda.min)

# elastic net 
out.en = glmnet(x=train.data, y=train.target, alpha = 0.5, family = "binomial", nlambda = 250)
plot(out.en$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(out.en$beta)
image(out.en$beta)
matplot(t(out.en$beta), type = "l")

#select lambda and find best cv en
set.seed(123)
cv.logistic.en.model <- cv.glmnet(dmx.train, dmy.train, family = "binomial", alpha = 0.5, standardize=TRUE)

lgrid = out.en$lambda
agrid = seq(0,1,length.out=25)

set.seed(123)
cvloop = cv.glmnet(dmx.train, dmy.train, family = "binomial", alpha=agrid[1], lambda = lgrid)
cvloop = cbind(rep(agrid[1],length(cvloop$lambda)),cvloop$lambda,cvloop$cvm)
for(i in 2:length(agrid)){
  res_i =   cv.glmnet(dmx.train, dmy.train, family = "binomial", alpha=agrid[i], lambda = lgrid)
  cvloop = rbind(cvloop, cbind(rep(agrid[i],length(res_i$lambda)),res_i$lambda,res_i$cvm))
}
cvloop = as.data.frame(cvloop)
names(cvloop) <- c("alpha","lambda","cvm")
library(ggplot2)
ggplot(cvloop, aes(x=lambda, y=cvm, group=alpha, color=alpha))+geom_line()
cvloop[which.min(cvloop$cvm),]
alphastar = cvloop[which.min(cvloop$cvm),1]
lambdastar = cvloop[which.min(cvloop$cvm),2]

en = glmnet(dmx.train, dmy.train, family = "binomial", alpha = alphastar, lambda = lambdastar)

# Cross-validation results for logistic models
print(logistic.model)
print(ridge)
print(lasso)
print(en)

#forecasts
pr.logistic.model <- predict(logistic.model, newdata=test.data, type="response")
pr.ridge <- predict.glmnet(ridge, newx= dmx.test, type="response")
pr.lasso <- predict.glmnet(lasso, newx= dmx.test, type="response")
pr.en <- predict.glmnet(en, newx= dmx.test, type="response")

# Assuming threshold is 0.5
threshold <- 0.5

# Convert predictions to binary class labels
predicted_labels_logistic <- as.integer(pr.logistic.model > threshold)
predicted_labels_ridge <- as.integer(pr.ridge > threshold)
predicted_labels_lasso <- as.integer(pr.lasso > threshold)
predicted_labels_en <- as.integer(pr.en > threshold)

# Calculating ROC and AUC
roc.logistic.model <- roc(test.target, predicted_labels_logistic)
roc.ridge <- roc(test.target, predicted_labels_ridge)
roc.lasso <- roc(test.target, predicted_labels_lasso)
roc.en <- roc(test.target, predicted_labels_en)

auc_logistic_model <- auc(roc(test.target, predicted_labels_logistic))
auc_ridge <- auc(roc(test.target, predicted_labels_ridge))
auc_lasso<- auc(roc(test.target, predicted_labels_lasso))
auc_en <- auc(roc(test.target, predicted_labels_en))

# Print AUC 
print(c(
  Logistic_Model = auc_logistic_model,
  CV_Ridge_Model = auc_ridge,
  CV_Lasso_Model = auc_lasso,
  CV_Elastic_Net_Model = auc_en
))

# Plot ROC curves
plot(roc.logistic.model, col = "blue", main = "ROC Curves for Logistic Regression Models")
plot(roc.ridge, col = "green", add = TRUE)
plot(roc.lasso, col = "orange", add = TRUE)
plot(roc.en, col = "purple", add = TRUE)

legend("bottomright", legend = c("Logistic Model", "CV Ridge Model", "CV Lasso Model", "CV Elastic Net Model"),
       col = c("blue", "green", "orange", "purple"), lty = 1)

text(0.8, 0.2, paste("AUC Logistic Model =", round(auc(roc.logistic.model), 3)), col = "blue")
text(0.8, 0.15, paste("AUC CV Ridge Model =", round(auc(roc.ridge), 3)), col = "green")
text(0.8, 0.1, paste("AUC CV Lasso Model =", round(auc(roc.lasso), 3)), col = "orange")
text(0.8, 0.05, paste("AUC CV Elastic Net Model =", round(auc(roc.en), 3)), col = "purple")

# AUC below 0.7, not a good result

# Confusion matrix 
confusion_logistic_model <- confusionMatrix(factor(round(predicted_labels_logistic)), factor(test.target), positive="1")
confusion_cv_logistic_ridge_model <- confusionMatrix(factor(round(predicted_labels_ridge)), factor(test.target),positive="1")
confusion_cv_logistic_lasso_model <- confusionMatrix(factor(round(predicted_labels_lasso)), factor(test.target),positive="1")
confusion_cv_logistic_en_model <- confusionMatrix(factor(round(predicted_labels_en)), factor(test.target),positive="1")

# Print confusion matrices
print(list(
  Logistic_Model = confusion_logistic_model,
  CV_Ridge_Model = confusion_cv_logistic_ridge_model,
  CV_Lasso_Model = confusion_cv_logistic_lasso_model,
  CV_Elastic_Net_Model = confusion_cv_logistic_en_model
))

# Define false positive and false negative costs
cost_fp <- 1  # Cost of false positive
cost_fn <- 10  # Cost of false negative

# Function to calculate total cost
calculate_total_cost <- function(predictions, true_labels, threshold) {
  predicted_labels <- ifelse(predictions > threshold, 1, 0)
  confusion_matrix <- confusionMatrix(factor(predicted_labels), factor(true_labels))
  total_cost <- cost_fp * confusion_matrix$table[1, 2] + cost_fn * confusion_matrix$table[2, 1]
  return(total_cost)
}

# Logistic Regression
total_cost_logistic_model <- calculate_total_cost(pr.logistic.model, test.target, threshold)
print(paste("Total Cost for Logistic Regression Model at Threshold 0.5:", total_cost_logistic_model))

# Ridge Regression
total_cost_ridge <- calculate_total_cost(pr.ridge, test.target, threshold)
print(paste("Total Cost for Ridge Model at Threshold 0.5:", total_cost_ridge))

# Lasso Regression
total_cost_lasso <- calculate_total_cost(pr.lasso, test.target, threshold)
print(paste("Total Cost for Lasso Model at Threshold 0.5:", total_cost_lasso))

# Elastic Net
total_cost_en <- calculate_total_cost(pr.en, test.target, threshold)
print(paste("Total Cost for Elastic Net Model at Threshold 0.5:", total_cost_en))

# Create a data frame with the results
results <- data.frame(
  Model = c("Logistic Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"),
  Total_Cost = c(total_cost_logistic_model, total_cost_ridge, total_cost_lasso, total_cost_en)
)

# Print the table
kable(results, format = "html", caption = "Total Costs for Different Models at Threshold 0.5") %>%
  kable_styling()

#ridge has the minimum cost (almost equal to elastic net)

#####################################
#####  K-NN CLASSIFIER ##############
#####################################
set.seed(123)

# Function to calculate error rate
calculate_error_rate <- function(predictions.knn, true_labels) {
  error_rate.knn <- sum(predictions.knn != true_labels) / length(true_labels)
  return(error_rate.knn)
}

# Values of k to try
k_values <- c(1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39)

# Initialize results storage
results.knn <- data.frame(k = k_values, Error_Rate = numeric(length(k_values)), AUC = numeric(length(k_values)), Total_Costs = numeric(length(k_values)))

# Initialize storage for confusion matrices
confusion_matrices <- list()

# Iterate over k values
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  # Train K-NN model
  knn_model <- knn(train.data, valid.data, train.target, k = k)
  
  # Make predictions
  predictions.knn <- knn_model
  
  # Calculate error rate
  error_rate.knn <- calculate_error_rate(predictions.knn, valid.target)
  
  # Store results
  results.knn[i, "Error_Rate"] <- error_rate.knn
  
  # Confusion matrix
  confusion.knn <- table(valid.target, predictions.knn)
  confusion_matrices[[paste("k", k)]] <- confusion.knn
  print(confusion.knn)
  
  # Plot
  plot(confusion.knn, main = paste("Confusion Matrix (k =", k, ")"))
  
  # Calculate AUC
  roc_knn <- roc(valid.target, as.numeric(predictions.knn))
  auc_knn <- auc(roc_knn)
  print(paste("AUC for k =", k, ":", round(auc_knn, 3)))
  
  #store AUC in results
  results.knn[i, "AUC"] <- auc_knn
  
  # Plot ROC curve
  plot(roc_knn, main = paste("ROC Curve for k =", k))
  
  # Plot Precision-Recall curve
  pr_curve_knn <- pr.curve(valid.target, as.numeric(predictions.knn),curve = TRUE)
  plot(pr_curve_knn, main = paste("Precision-Recall Curve for k =", k))
  
  #calculate total costs
  total_cost_knn <- cost_fp * confusion.knn[1, 2] + cost_fn * confusion.knn[2, 1]
  
  #store Total Costs in results
  results.knn[i, "Total costs"] <- total_cost_knn
}

# Print results
print(results.knn)

# Find the value of k with the maximum AUC, minimum error and minimum cost
best_k_AUC <- results.knn[which.max(results.knn$AUC), "k"]
best_k_error <- results.knn[which.min(results.knn$Error_Rate), "k"]
best_k_costs <- results.knn[which.min(results.knn$`Total costs`), "k"]

# Print the best k values
print(paste("Best k based on Error Rate:", best_k_error))
print(paste("Best k based on AUC:", best_k_AUC))
print(paste("Best k based on Costs:", best_k_costs))

# Fit the final k-NN model on the training set with the best value of k
final_knn_model.1 <- knn(train.data, test.data, train.target, k = best_k_error)
final_knn_model.2 <- knn(train.data, test.data, train.target, k = best_k_AUC)
final_knn_model.3 <- knn(train.data, test.data, train.target, k = best_k_costs)

# Make predictions on the test set
predictions_knn_test.1 <- final_knn_model.1
predictions_knn_test.2 <- final_knn_model.2
predictions_knn_test.3 <- final_knn_model.3

# Error rate
error_rate_final_knn_model.1 <- calculate_error_rate(predictions_knn_test.1, test.target)
error_rate_final_knn_model.2 <- calculate_error_rate(predictions_knn_test.2, test.target)
error_rate_final_knn_model.3 <- calculate_error_rate(predictions_knn_test.3, test.target)

# Print error rates
cat("Error Rate for Final k-NN Model (Best k=35 based on Error Rate):", round(error_rate_final_knn_model.1, 3), "\n")
cat("Error Rate for Final k-NN Model (Best k=3 based on AUC):", round(error_rate_final_knn_model.2, 3), "\n")
cat("Error Rate for Final k-NN Model (Best k=3 based on AUC):", round(error_rate_final_knn_model.3, 3), "\n")

#evaluating model performance
#accuracy    
accuracy.1 <- sum(predictions_knn_test.1 == test.target) / length(test.target)
print(paste("Accuracy with k=35:", round(accuracy.1, 3)))     

accuracy.2 <- sum(predictions_knn_test.2 == test.target) / length(test.target)
print(paste("Accuracy with k=3:", round(accuracy.2, 3)))     

accuracy.3 <- sum(predictions_knn_test.3 == test.target) / length(test.target)
print(paste("Accuracy with k=3:", round(accuracy.3, 3)))     

#accuracy=0.686, 0.683 and 0.613, not that good. is dataframe not well balanced?

table(target)
# Calculate percentage of instances in the minority class
percentage_minority <- sum(target == 1) / nrow(data) * 100
print(paste("Percentage of Minority Class:", percentage_minority, "%"))
# Plot class distribution
barplot(table(target), main = "Class Distribution", col = c("red", "blue"), legend = TRUE)
# Calculate imbalance ratio
imbalance_ratio <- table(target)[[2]] / table(target)[[1]]
print(paste("Imbalance Ratio:", imbalance_ratio))

# yes, df is not balanced,using different metrics for evaluation

#precision, recall, F1-score
#confusion matrix 1
confusion.final_knn.1 <- table(test.target, predictions_knn_test.1)
confusion_matrices[[paste("k", k)]] <- confusion.final_knn.1
print(confusion.final_knn.1)

#metrics 1
true_positive.1 <- confusion.final_knn.1[2, 2]
precision.1 <- true_positive.1 / sum(confusion.final_knn.1[, 2])
recall.1 <- true_positive.1 / sum(confusion.final_knn.1[2, ])
f1_score.1 <- 2 * (precision.1 * recall.1) / (precision.1 + recall.1)

#print results 1
print(paste("Precision for k=35:", round(precision.1, 3)))
print(paste("Recall for k=35:", round(recall.1, 3)))
print(paste("F1 Score for k=35:", round(f1_score.1, 3)))

#confusion matrix 2
confusion.final_knn.2 <- table(test.target, predictions_knn_test.2)
confusion_matrices[[paste("k", k)]] <- confusion.final_knn.2
print(confusion.final_knn.2)

#metrics 2
true_positive.2 <- confusion.final_knn.2[2, 2]
precision.2 <- true_positive.2 / sum(confusion.final_knn.1[, 2])
recall.2 <- true_positive.2 / sum(confusion.final_knn.2[2, ])
f1_score.2 <- 2 * (precision.2 * recall.2) / (precision.2 + recall.2)

#print results 2
print(paste("Precision for k=3:", round(precision.2, 3)))
print(paste("Recall for k=3:", round(recall.2, 3)))
print(paste("F1 Score for k=3:", round(f1_score.2, 3)))

#confusion matrix 3
confusion.final_knn.3 <- table(test.target, predictions_knn_test.3)
confusion_matrices[[paste("k", k)]] <- confusion.final_knn.3
print(confusion.final_knn.3)

#metrics 1
true_positive.3 <- confusion.final_knn.3[2, 2]
precision.3 <- true_positive.3 / sum(confusion.final_knn.3[, 2])
recall.3 <- true_positive.3 / sum(confusion.final_knn.3[2, ])
f1_score.3 <- 2 * (precision.3 * recall.3) / (precision.3 + recall.3)

#print results 3
print(paste("Precision for k=39:", round(precision.3, 3)))
print(paste("Recall for k=39:", round(recall.3, 3)))
print(paste("F1 Score for k=39:", round(f1_score.3, 3)))

# best k-nn are k=3 and k=35, but costs-wise k=39 is to be preferred

#####################################
#####  RANDOM FOREST ################
#####################################

set.seed(123)

# Define the range for the number of trees
ntree_range <- seq(500, 2000, by = 250)

# Initialize results storage for the number of trees
results_rf_ntree <- data.frame(ntree = numeric(length(ntree_range)), AUC = numeric(length(ntree_range)))

# Hyperparameter tuning for the number of trees
for (i in seq_along(ntree_range)) {
  # Fit Random Forest model on the training set
  rf_model <- randomForest(as.factor(train.target) ~ ., data = train.data, ntree = ntree_range[i])
  
  # Make predictions on the validation set
  predictions_rf_val <- predict(rf_model, newdata = valid.data)
  
  # Evaluate Random Forest model on the validation set
  roc_rf_val <- roc(valid.target, as.numeric(predictions_rf_val))
  auc_rf_val <- auc(roc_rf_val)
  
  # Store results
  results_rf_ntree[i, "ntree"] <- ntree_range[i]
  results_rf_ntree[i, "AUC"] <- auc_rf_val
}

# Print results
print(results_rf_ntree)

# Choose the optimal number of trees (e.g., the one with the highest AUC)
best_ntree <- results_rf_ntree[which.max(results_rf_ntree$AUC), "ntree"]

# Define the range for the number of variables at each split
mtry_range <- seq(1, ncol(train.data)-1, by = 1)

# Initialize results storage for the number of variables at each split
results_rf_mtry <- data.frame(mtry = numeric(length(mtry_range)), AUC = numeric(length(mtry_range)))

# Hyperparameter tuning for the number of variables at each split
for (j in seq_along(mtry_range)) {
  # Fit Random Forest model on the training set
  rf_model <- randomForest(as.factor(train.target) ~ ., data = train.data, ntree = best_ntree, mtry = mtry_range[j])
  
  # Make predictions on the validation set
  predictions_rf_val <- predict(rf_model, newdata = valid.data)
  
  # Evaluate Random Forest model on the validation set
  roc_rf_val <- roc(valid.target, as.numeric(predictions_rf_val))
  auc_rf_val <- auc(roc_rf_val)
  
  # Store results
  results_rf_mtry[j, "mtry"] <- mtry_range[j]
  results_rf_mtry[j, "AUC"] <- auc_rf_val
}

# Print results
print(results_rf_mtry)

# Choose the optimal number of variables at each split (the one with the highest AUC)
best_mtry <- results_rf_mtry[which.max(results_rf_mtry$AUC), "mtry"]

# Train the final Random Forest model on the training set with the optimal hyperparameters
final_rf_model <- randomForest(as.factor(train.target) ~ ., data = train.data, ntree = 1000, mtry = 12)

# Make predictions on the test set
predictions_rf_test <- predict(final_rf_model, newdata = test.data)

# Evaluate Random Forest model on the test set
roc_rf_test <- roc(test.target, as.numeric(predictions_rf_test))
auc_rf_test <- auc(roc_rf_test)

# Print final AUC on the test set
print(paste("Final AUC on Test Set:", round(auc_rf_test, 3)))

#confusion matrix rf
confusion.final_rf <- table(test.target, predictions_rf_test)
confusion_matrices[[paste("k", k)]] <- confusion.final_rf
print(confusion.final_rf)

#total costs rf
total_cost_rf <- cost_fp * confusion.final_rf[1, 2] + cost_fn * confusion.final_rf[2, 1]

print(total_cost_rf)

#metrics 
accuracy.rf <- sum(predictions_rf_test == test.target) / length(test.target)
print(paste("Accuracy with trees=1000 and variables=12:", round(accuracy.rf, 3)))     

true_positive.rf <- confusion.final_rf[2, 2]
precision.rf <- true_positive.rf / sum(confusion.final_rf[, 2])
recall.rf <- true_positive.rf / sum(confusion.final_rf[2, ])
f1_score.rf <- 2 * (precision.rf * recall.rf) / (precision.rf + recall.rf)

#print results 3
print(paste("Precision for rf:", round(precision.rf, 3)))
print(paste("Recall for rf:", round(recall.rf, 3)))
print(paste("F1 Score for rf:", round(f1_score.rf, 3)))


#####################################
#####  NEURAL NETWORK  ##############
#####################################

# prepare data for the model
x          <- data[, unlist(lapply(data, is.numeric))]
x          <- scale(x) # recommended with networks 
y          <- data[, 61]
test      <- sample(1:nrow(data)*0.1)
train      <- -test 
x_train    <- x[train,c(3, 4, 8, 10:19, 39, 45:48)]
x_test     <- x[test,]
y_train    <- y[train]
y_test     <- y[test]
df_train   <- data.frame(x_train, y_train)
df_test    <- data.frame(x_test, y_test)
model_list <- "y_train ~ ." 

# build model
hidden    <- c(15,9,6,3)
threshold <- 1.5
stepmax   <- 5000 
rep       <- 3
model     <- neuralnet(model_list, data = df_train, hidden = hidden, threshold = threshold,
                       stepmax = stepmax, rep = rep, lifesign = "minimal", 
                       act.fct = "logistic", linear.output = FALSE)
plot(model)

# look at predictions and confusion matrix
thr        <- 0.5
pred_train <- ifelse(model$net.result[[1]][,1] > thr, "no", "yes")
tb_train   <- table(y_train, pred_train)
tb_train
1-sum(diag(tb_train))/sum(tb_train)

out_test  <- compute(model, df_test)$net.result
pred_test <- ifelse(out_test[,1] > thr, "no", "yes")
tb_test   <- table(y_test, pred_test)
tb_test
1-sum(diag(tb_test))/sum(tb_test)

#### shallow network 

hidden    <- hidden <- 5
threshold <- 1.2
stepmax   <- 1500 
rep       <- 1
model.shallow     <- neuralnet(model_list, data = df_train, hidden = hidden, threshold = threshold,
                               stepmax = stepmax, rep = rep, lifesign = "minimal", 
                               act.fct = "logistic", linear.output = FALSE)

thr        <- 0.5
pred_train <- ifelse(model.shallow$net.result[[1]][,1] > thr, "no", "yes")
tb_train   <- table(y_train, pred_train)
tb_train
1-sum(diag(tb_train))/sum(tb_train)

out_test_shallow  <- compute(model.shallow, df_test)$net.result
pred_test <- ifelse(out_test[,1] > thr, "no", "yes")
tb_test   <- table(y_test, pred_test)
tb_test
1-sum(diag(tb_test))/sum(tb_test)

#### deep network 

hidden    <- hidden <- c(15, 10, 5, 3)
threshold <- 0.001
stepmax   <- 3000 
rep       <- 5
model.deep     <- neuralnet(model_list, data = df_train, hidden = hidden, threshold = threshold,
                            stepmax = stepmax, rep = rep, lifesign = "minimal", 
                            act.fct = "logistic", linear.output = FALSE)

plot(model.deep)
thr        <- 0.5
pred_train <- ifelse(model.deep$net.result[[1]][,1] > thr, "no", "yes")
tb_train   <- table(y_train, pred_train)
tb_train
1-sum(diag(tb_train))/sum(tb_train)

out_test_deep  <- compute(model.shallow, df_test)$net.result
pred_test_deep <- ifelse(out_test[,1] > thr, "no", "yes")
tb_test_deep   <- table(y_test, pred_test_deep)
tb_test_deep
1-sum(diag(tb_test_deep))/sum(tb_test)

# evaluate deep model
thr <- 0.5
pred_train_deep <- ifelse(model.deep$net.result[[1]][, 1] > thr, "no", "yes")
tb_train_deep <- table(y_train, pred_train_deep)
accuracy_train_deep <- 1 - sum(diag(tb_train_deep)) / sum(tb_train_deep)
cat("Training Accuracy (Deep Model):", accuracy_train_deep, "\n")

out_test_deep <- compute(model.deep, df_test)$net.result
pred_test_deep <- ifelse(out_test_deep[, 1] > thr, "no", "yes")
tb_test_deep <- table(y_test, pred_test_deep)
accuracy_test_deep <- 1 - sum(diag(tb_test_deep)) / sum(tb_test_deep)
cat("Testing Accuracy (Deep Model):", accuracy_test_deep, "\n")


#####################################
######## RESULTS  ###################
#####################################


#the best model is the Random Forest model with 1000 trees and 12 variables, which has a lower total cost (4304) 
#in respect to the other models analysed, with an accuracy of 0.693, a 
#precision of 0.732, recall of 0.874 and a f1-score of 0.797, The expected misclassification error is 0.309.
