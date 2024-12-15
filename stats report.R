### PART I ###
data = read.csv("~/Downloads/DSA1101/Data/diabetes_5050.csv")
head(data)
dim(data)
set.seed(1101)

## EXTRACT COLUMNS OF DATA ## 
# CATEGORICAL #
Diabetes_binary = as.factor(data$Diabetes_binary)
HighBP = as.factor(data$HighBP)
HighChol = as.factor(data$HighChol)
CholCheck = as.factor(data$CholCheck)
Smoker = as.factor(data$Smoker)
Stroke = as.factor(data$Stroke)
HeartDiseaseorAttack = as.factor(data$HeartDiseaseorAttack)
PhysActivity = as.factor(data$PhysActivity)
Fruits = as.factor(data$Fruits)
Veggies = as.factor(data$Veggies)
HvyAlcoholConsump = as.factor(data$HvyAlcoholConsump)
AnyHealthcare = as.factor(data$AnyHealthcare)
NoDocbcCost = as.factor(data$NoDocbcCost)
DiffWalk = as.factor(data$DiffWalk)
Sex = as.factor(data$Sex)

# QUANTITATIVE #
BMI = data$BMI
MentHlth = data$MentHlth
PhysHlth = data$PhysHlth
GenHlth = data$GenHlth
Age = data$Age
Education = data$Education
Income = data$Income

## FINDING ASSOCATION BETWEEN VARIABLES ##
## USING BOX PLOTS (1 Categorical & 1 Quantitative Variable) ## 
# BMI #
boxplot(BMI ~ Diabetes_binary, col = "red")
# MENTAL HEALTH #
boxplot(MentHlth ~ Diabetes_binary, col = "pink")
# PHYSICAL HEALTH # 
boxplot(PhysHlth ~ Diabetes_binary, col = "blue")
# GENERAL HEALTH #
boxplot(GenHlth ~ Diabetes_binary, col = "green")
# AGE #
boxplot(Age ~ Diabetes_binary, col = "purple")
# EDUCATION #
boxplot(Education ~ Diabetes_binary, col = "yellow")
# INCOME #
boxplot(Income ~ Diabetes_binary, col = "orange")

## USING ODDS RATIO (2 Categorical Variables) ##
odds_ratio <- function(x) {
  odds.ratio = (x[1,1]*x[2,2]) / (x[1,2]*x[2,1]) 
  return(odds.ratio)
}
# HIGH BLOOD PRESSURE #
T1 = table(HighBP, Diabetes_binary)
odds_ratio(T1) # 5.088477
# HIGH CHOLESTRAL #
T2 = table(HighChol, Diabetes_binary)
odds_ratio(T2) # 3.296316
# CHOLESTRAL CHECK #
T3 = table(CholCheck, Diabetes_binary)
odds_ratio(T3) # 6.491553
# SMOKER #
T4 = table(Smoker, Diabetes_binary)
odds_ratio(T4) # 1.412383
# STROKE #
T5 = table(Stroke, Diabetes_binary)
odds_ratio(T5) # 3.093272
# HEART DISEASE OR ATTACK #
T6 = table(HeartDiseaseorAttack, Diabetes_binary)
odds_ratio(T6) # 3.656197
# PHYSICAL ACTIVITY #
T7 = table(PhysActivity, Diabetes_binary)
odds_ratio(T7) # 0.4939616
# FRUITS #
T8 = table(Fruits, Diabetes_binary)
odds_ratio(T8) # 0.8007655
# VEGGIES #
T9 = table(Veggies, Diabetes_binary)
odds_ratio(T9) # 0.6763796
# HEAVY ALCOHOL CONSUMPTION #
T10 = table(HvyAlcoholConsump, Diabetes_binary) 
odds_ratio(T10) # 0.3653163
# ANY HEALTHCARE #
T11 = table(AnyHealthcare, Diabetes_binary) 
odds_ratio(T11) # 1.251644
# NO DOCTOR BC COST #
T12 = table(NoDocbcCost, Diabetes_binary)
odds_ratio(T12) # 1.326217
# DIFFICULTY IN WALKING #
T13 = table(DiffWalk, Diabetes_binary)
odds_ratio(T13) # 3.807365
# GENDER #
T14 = table(Sex, Diabetes_binary)
odds_ratio(T14) # 1.195343

## REMOVE THE VARIABLE EDUCATION ##
data = data[,-21]

## SPLIT INTO TRAIN & TEST DATA ##
# total number of rows = 70692
disease = which(data[,1]==1)
no_disease = which(data[,1]!=1)
n_folds = 5
folds_j <- c(sample(rep(1:n_folds, length.out = length(disease))), 
             rep(1:n_folds, length.out = length(no_disease)))
auc = numeric(n_folds)
err = numeric(n_folds)
acc = numeric(n_folds)
tpr = numeric(n_folds)
fpr = numeric(n_folds)
fnr = numeric(n_folds)
prec = numeric(n_folds)

### PART II ###
library(ROCR)
## DECISION TREE ##
library("rpart")
library("rpart.plot")

for (j in 1:n_folds) {
  test_j <- which(folds_j == j)
  fit <- rpart(Diabetes_binary ~., method="class", data = data[-test_j,],
               control=rpart.control(minsplit=20), parms=list(split='gini')
  ) 
  rpart.plot(fit, type=4, extra=2, varlen=0, faclen=0)
  dt_Y = predict(fit, newdata=data[test_j,-1], type="class")
  dt_Y = as.numeric(as.character(dt_Y))
  
  pred = prediction(dt_Y, data[test_j,1]) 
  perf = performance(pred, "tpr", "fpr")
  auc[j] = performance(pred, "auc")@y.values[[1]]
  
  t = table(data[test_j,1], dt_Y)
  err[j]=mean(data[test_j,1] != dt_Y)
  acc[j]=mean(data[test_j,1] == dt_Y)
  tpr[j] = t[1,1]/(t[1,1] + t[1,2])
  fpr[j] = t[2,1]/(t[2,1] + t[2,2])
  fnr[j] = t[1,2]/(t[1,1] + t[1,2])
  prec[j] = t[1,1]/(t[1,1] + t[2,1])
}
mean(acc)
mean(err)
mean(tpr)
mean(fpr)
mean(fnr)
mean(auc)
mean(prec)

## NAIVE BAYES ## 
library(e1071)

for (j in 1:n_folds) {
  test_j <- which(folds_j == j)
  model <- naiveBayes(Diabetes_binary ~ ., data = data[-test_j,])
  nb_Y = predict(model, data[test_j, -1], "class")
  nb_Y = as.numeric(as.character(nb_Y))
  
  pred = prediction(nb_Y, data[test_j,1]) 
  perf = performance(pred, "tpr", "fpr")
  auc[j] = performance(pred, "auc")@y.values[[1]]
  
  t = table(data[test_j,1], nb_Y)
  err[j] = mean(data[test_j,1] != nb_Y)
  acc[j] = mean(data[test_j,1] == nb_Y)
  tpr[j] = t[1,1]/(t[1,1] + t[1,2])
  fpr[j] = t[2,1]/(t[2,1] + t[2,2])
  fnr[j] = t[1,2]/(t[1,1] + t[1,2])
  prec[j] = t[1,1]/(t[1,1] + t[2,1])
}
mean(acc)
mean(err)
mean(tpr)
mean(fpr)
mean(fnr)
mean(auc)
mean(prec)

## LOGISTIC REGRESSION ##
for (j in 1:n_folds) {
  threshold = 0.482
  test_j <- which(folds_j == j)
  M1 <- glm(Diabetes_binary ~., data = data[-test_j,], 
            family = binomial(link = "logit"))
  lr_Y = predict(M1, newdata = data[test_j,-1], type = "response")
  lr_Y = ifelse(lr_Y > threshold, 1, 0)
  lr_Y = as.numeric(as.character(lr_Y))

  pred = prediction(lr_Y, data[test_j,1]) 
  perf = performance(pred, "tpr", "fpr")
  auc[j] = performance(pred, "auc")@y.values[[1]]
  
  t = table(data[test_j,1], lr_Y)
  err[j] = mean(data[test_j,1] != lr_Y)
  acc[j] = mean(data[test_j,1] == lr_Y)
  tpr[j] = t[1,1]/(t[1,1] + t[1,2])
  fpr[j] = t[2,1]/(t[2,1] + t[2,2])
  fnr[j] = t[1,2]/(t[1,1] + t[1,2])
  prec[j] = t[1,1]/(t[1,1] + t[2,1])
}

mean(acc) 
mean(err) 
mean(tpr) 
mean(fpr) 
mean(fnr) 
mean(auc) 
mean(prec)

## FURTHER COMMENTS ON LOGISTIC REGRESSION ##
M2 <- glm(Diabetes_binary ~., data = data, 
          family = binomial(link = "logit"))
summary(M2)

new_data = data[,c(-6,-9,-10,-11,-13,-14,-16)]
new_model <- glm(Diabetes_binary ~., data = new_data, 
          family = binomial(link = "logit"))
summary(new_model)
head(new_data)
dim(new_data)

disease = which(new_data[,1]==1)
no_disease = which(new_data[,1]!=1)
n_folds = 5
folds_j <- c(sample(rep(1:n_folds, length.out = length(disease))), 
             rep(1:n_folds, length.out = length(no_disease)))

for (j in 1:n_folds) {
  threshold = 0.32
  test_j <- which(folds_j == j)
  M3 <- glm(Diabetes_binary ~., data = new_data[-test_j,], 
            family = binomial(link = "logit"))
  lr_Y = predict(M3, newdata = new_data[test_j,-1], type = "response")
  lr_Y = ifelse(lr_Y > threshold, 1, 0)
  lr_Y = as.numeric(as.character(lr_Y))
  
  pred = prediction(lr_Y, new_data[test_j,1]) 
  perf = performance(pred, "tpr", "fpr")
  auc[j] = performance(pred, "auc")@y.values[[1]]
  
  t = table(new_data[test_j,1], lr_Y)
  err[j] = mean(new_data[test_j,1] != lr_Y)
  acc[j] = mean(new_data[test_j,1] == lr_Y)
  tpr[j] = t[1,1]/(t[1,1] + t[1,2])
  fpr[j] = t[2,1]/(t[2,1] + t[2,2])
  fnr[j] = t[1,2]/(t[1,1] + t[1,2])
  prec[j] = t[1,1]/(t[1,1] + t[2,1])
}

mean(acc) 
mean(err) 
mean(tpr) 
mean(fpr) 
mean(fnr) 
mean(auc) 
mean(prec)









