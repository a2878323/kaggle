
#install.packages("ggplot2")
library('ggplot2') # visualization
#install.packages("ggthemes")
library('ggthemes') # visualization
#install.packages("scales")
library('scales') # visualization
#install.packages("dplyr")
library('dplyr') # data manipulation
#install.packages("mice")
library('mice') # imputation
#install.packages("randomForest")
library('randomForest') # classification algorithm
library('rpart') #decesion tree
#install.packages("nnet")
#install.packages("neuralnet")
library('nnet') #類神經網路
library("neuralnet")
#install.packages("caret")
library(caret) #train
library(class) #train
#install.packages("e1071") #SVM
library(e1071)
library(caTools)
library(gridExtra)
library(VGAM)
library(gbm) #GBM
library(kernlab) #SVM
#install.packages("klaR")
library(klaR)

#讀取資料
rm(list = ls())
setwd("~/Documents/Kaggle/Titanic")
train=read.csv("train.csv",header=T,stringsAsFactors = F)
test=read.csv("test.csv",header=T,stringsAsFactors = F)
train$flag="train"
test$flag="test"
all=bind_rows(train,test)

#資料型態
str(all) 
summary(all)

#檢查是否有遺失值
md.pattern(all)  #連續形變數是否有遺失值

# Extract Family Name 抓取家族姓氏 
full$FamilyN=sapply(full$Name,function(x) strsplit(x,"[,.]")[[1]][1])

# training control for caret 不懂，不懂參數要怎麼放
trControl <- trainControl(method="repeatedcv", number=7, repeats=5);

#家族人數
full$Fsize <- full$SibSp + full$Parch + 1

#家族人數分群
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

#家族姓氏
full$Surname <- sapply(full$Name,function(x) strsplit(x, split = '[,.]')[[1]][1])

#補Fare遺失值
# Predict missing fares only 1044
faremiss <- which(is.na(full$Fare));

# Create a fare factor
full$FareFac <- factor(full$Fare);
# 用決策樹補
modelf   <- train( FareFac ~ Pclass + Sex + Embarked + SibSp + Parch, data = full, trControl = trControl, method="rpart", na.action = na.pass);

full$FareFac[faremiss] = predict(modelf, full[faremiss,]); 
full$Fare[faremiss] <- as.numeric(as.character(full$FareFac[faremiss]));

#補embarked 遺失值
# Predict missing embarked entries
embarkmiss <- which(full$Embarked == ''); 
modelp <- train( Embarked ~ Pclass + Sex + Fare, data = full, trControl = trControl, method="rpart", na.action = na.pass);
full$Embarked[embarkmiss] = predict(modelp, full[embarkmiss, ]); 

# Title
# Extract titles. There are many but we will consolidate to four of them: Minor, Miss, Mrs, Mr.
full$Title <- sapply(as.character(full$Name), FUN=function(x) {trimws(strsplit(x, split='[,.]')[[1]][2])});

full$Title[full$Title=='Master'] <- 'Minor'
full$Title[full$Title == 'Dr'] <- 'Mr'; 
full$Title[797] <- 'Miss'; # only one female doctor 

full$Title[full$Title  %in% c('Ms',  'Mme', 'Mlle', 'the Countess')] <- 'Miss';
full$Title[full$Title  %in% c('Dona', 'Lady')] <- 'Mrs';
full$Title[!full$Title %in% c('Minor', 'Mrs', 'Miss')] <- 'Mr'; # Assign any other remaining titles to Mr, ensuring we have only four distinct titles.



#補年齡遺失值
factor_vars <- c('PassengerId','Pclass','Sex','Embarked','Title','Surname','FsizeD')
full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))
#使用mice 來補age
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','flag','FarFac','Surname','Survived')], method='rf')
mice_output <- complete(mice_mod)

#確認補值的分佈
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

#分佈差不多，補回原本資料
full$Age <- mice_output$Age


# Compute Ticket and Fare frequencies 
#計算同樣的票有幾個人
full$TFreq <- ave(seq(nrow(full)), full$Ticket,  FUN=length);
#同樣票價的有幾人
full$FFreq <- ave(seq(nrow(full)), full$FareFac, FUN=length);
#同樣艙房的有幾人
full$CFreq <- ave(seq(nrow(full)), full$Cabin,   FUN=length);


par(mfrow=c(1,1))


# Engineer group Ids (GID) using SibSp, Parch, TFreq, and FFreq to label groups
full$GID <- rep(NA, nrow(full));
maxgroup <- 12; # maximum size of a group
for ( i in as.numeric(full$PassengerId)) {
  if(full$SibSp[i] + full$Parch[i] > 0) { # Check first if ID has relatives #姓名與攜帶眷屬人數一組
    full$GID[i] <- paste0(full$Surname[i], full$SibSp[i] + full$Parch[i]);
  } else {
    if(full$TFreq[i] > 1 & is.na(full$GID[i])) { # Next if shares ticket number with others 同票的一組 
      full$GID[i] <- as.character(full$Ticket[i]);
    } else {
      if(full$CFreq[i] > 1 & full$CFreq[i]<maxgroup & is.na(full$GID[i])) { #sharw Canin with others 同艙房的一組
        full$GID[i] <- as.character(full$Cabin[i]);
      }
      else {
        if(full$FFreq[i] > 1 & full$FFreq[i]<maxgroup & is.na(full$GID[i])) { # Next if shares Fare with others 票金額少且金額相同的一組
          full$GID[i] <- as.character(full$FareFac[i]);
        } else { 
          full$GID[i] <- "Single"; 
        }
      }
    }	
  }
}
full$GID <- factor(full$GID);

# Calculate fare per person for those with same ticket numbers
full$FareScaled <- full$Fare/full$TFreq;

# Transform to factors
full$Survived <- factor(full$Survived);

# Age processing is only done here by identifying Minors. Other than that Age will be ignored. 
# Everybody below a threshold age set to Minor.
child <- 14;
full$Title[full$Age<child] <- 'Minor'; 
full$Title[full$Age>=child & full$Sex=="male"] <- 'Mr';

# The next plot shows the fate of passengers in class 2 is almost certain
# Later will compute the log likelihood ratio for each of the 12 areas in the grid
test_index <- seq(418)+891;

p <- list();
item <- 1;
ylim <- 300;
for(title in c("Mr", "Mrs", "Miss", "Minor")) {
  for(class in c(1:3)){
    p[[item]] <- ggplot(full %>% filter(flag=="train", Pclass==class, Title==title), aes(x=Survived)) + geom_bar(aes(fill=Survived)) + scale_y_continuous(limits=c(0,ylim)) + theme(legend.position="none") + labs(title=paste('Pclass=', as.character(class), title));
    item <- item + 1;
  }
}
print(do.call(grid.arrange, p))

# define function to compute log likelihood of a/(1-a)
logl <- function(a) {
  a <- max(a,0.001); # avoids log(0)
  a <- min(a,0.990); # avoids division by 0
  return (log(a/(1-a)));
}

full$SLogL <- rep(0,nrow(full));

# Calculate the log likelihood ratio of survival probability as function of title and pclass
for (title in c("Mr", "Mrs", "Miss", "Minor")) {
  for (class in c(1:3)) {
    full$SLogL[full$Title==title& full$Pclass==class] <- logl(nrow(full %>% filter(Survived==1, Title==title, Pclass==class))/nrow(full %>% filter(Set=="train", Title==title, Pclass==class)));
  }
}

# This plot confirms there is information about survival by looking at the frequency of dupplicate tickets and fares
plot_fare <- ggplot(full %>% filter(flag=="train"), aes(x=FFreq, y=TFreq, color=Survived)) + geom_count(position=position_dodge(width=5)) + labs(title="Ticket and Fare Frequencies");
print(plot_fare)
plot_fare_density <- ggplot(full %>% filter(Set=="train"), aes(x=FFreq, y=TFreq, color=Survived)) + geom_density_2d() + labs(title="Ticket Frequency and Fare Frequency Density");
print(plot_fare_density)

# Next we reward or penalize groups of people. 
ticket_stats <- full %>% group_by(Ticket) %>% summarize(l = length(Survived), na = sum(is.na(Survived)), c = sum(as.numeric(Survived)-1, na.rm=T));

# 1) By incrementing the log likelihood score for groups larger than one that have survivors.
# Note we apply this bias only to groups that contain indivuals we need to predict.
# Applying to all seems to add noise. It is something that's worth more investigation.
for ( i in 1:nrow(ticket_stats)) {
  plist <- which(full$Ticket==ticket_stats$Ticket[i]);
  if(ticket_stats$na[i] > 0 & ticket_stats$l[i] > 1 & ticket_stats$c[i]>0) {
    full$SLogL[plist] <- full$SLogL[plist] + 3;
  }
}

# 2) By penalizing singles (See for example TFreq vs FFreq graph, bottom row represent singles)
full$SLogL[full$GID=="Single"] <- full$SLogL[full$GID=="Single"] - 2;

a=nrow(full %>% filter(Survived==1,GID=="Single"))/nrow(full %>% filter(GID=="Single"))
b=nrow(full %>% filter(Survived==1,GID!="Single"))/nrow(full %>% filter(GID!="Single"))


# 3) By penalizing large group sizes (See TFreq vs Pclass vs SLogL graph)
full$SLogL[full$TFreq ==  7] <- full$SLogL[full$TFreq == 7] - 3;
full$SLogL[full$TFreq ==  8] <- full$SLogL[full$TFreq == 8] - 1;
full$SLogL[full$TFreq == 11] <- full$SLogL[full$TFreq == 11] - 3;


a=nrow(full %>% filter(Survived==1,TFreq==11))/nrow(full %>% filter(TFreq==11))
b=nrow(full %>% filter(Survived==1,TFreq<7))/nrow(full %>% filter(TFreq<7))


# TFreq vs Pclass vs SLogL graph
# this graph was used to tune item 3) above
plot_slogl <- ggplot(full %>% filter(Set=="train"), aes(x=Pclass, y=SLogL)) + geom_jitter(aes(color=Survived)) + facet_grid(  . ~ TFreq,  labeller=label_both) + labs(title="SLogL vs Pclass vs TFreq")
print(plot_slogl)

# The last modeling and predict uses SLogL as the sole feature.
# This should prevent a lot of potential over-fitting.
fms <- formula("Survived ~ SLogL"); 

set.seed(2017);
knn_model <- train(fms, data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = "knn"); 

set.seed(2017);
gbm_model <- train(fms, data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = "gbm"); 

set.seed(2017);
rf_model <- train(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                    Fare + Embarked + Title + 
                    FsizeD
                  , data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = "rf"); 
set.seed(2017);
rf2_model <- train(factor(Survived) ~ Pclass + Sex + Age + SibSp +
                    Fare + Title + 
                    FsizeD+TFreq+CFreq+FFreq
                  , data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = "rf"); 

set.seed(2017)
rf_tune <- train(fms, data = full %>% filter(flag=="train"),method = "rf", trControl = trainControl(method = "boot632", number = 10) )

set.seed(2017);
dtree_model <- train(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                    Fare + Embarked + Title + 
                    FsizeD
                  , data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = "rpart"); 

full$Survived_num=as.numeric(full$Survived)-1
dummy=dummyVars( ~ Survived_num +Pclass + Sex + Age + SibSp + Parch + 
                  Fare + Embarked + Title + 
                  FsizeD,data = full %>% filter(flag=="train"))
dummy_data=predict(dummy,full %>% filter(flag=="train"))

dummy_test=dummyVars( ~ Survived_num +Pclass + Sex + Age + SibSp + Parch + 
                   Fare + Embarked + Title + 
                   FsizeD,data = full %>% filter(flag=="test"))
dummy_testdata=predict(dummy_test,full %>% filter(flag=="test"))


#set.seed(2017);
#nnet_model <- train(factor(Survived_num) ~ . , data = dummy_data, metric="Accuracy", trControl = trControl, method = "nnet"); 
set.seed(2017);
knn2_model <- train(factor(Survived_num) ~ . , data = dummy_data, metric="Accuracy", trControl = trControl, method = "knn"); 


set.seed(2017);
nnet_model <- train(fms , data=full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = "nnet"); 


set.seed(2017)
svm_model <- train(fms
                   , data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = 'svmRadial'); 



set.seed(2017)
nb_model <- train(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                     Fare + Embarked + Title + 
                     FsizeD
                   , data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = 'nb'); 

set.seed(2017)
nb2_model <- train(fms
                  , data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = 'nb'); 

#set.seed(2017);
#logit_model <- train(formula = factor(Survived) ~ Pclass + Sex + Age + Fare + 
#                      Title + FsizeD, data = full %>% filter(flag=="train"), metric="Accuracy", trControl = trControl, method = "vglmAdjCat"); 


# Using resamples function from the caret package to summarize the data
modelsummary <- resamples(list(knn=knn_model,knn2=knn2_model,rf1=rf_model,rf2=rf2_model,gbm=gbm_model
                               ,nnet=nnet_model,svm=svm_model,cart=dtree_model,
                               nb=nb_model,nb2=nb2_model))

# In-sample accuracy values for each model
summary(modelsummary)$statistics$Accuracy



sample_predictions <- 
  data.frame(RF_tune = predict(rf_tune,full %>% filter(flag=="train")), knn = predict(knn_model, full %>% filter(flag=="train")),
             GBM = predict(gbm_model,full %>% filter(flag=="train")), svm = predict(svm_model, full %>% filter(flag=="train")), nnet = predict(nnet_model, full %>% filter(flag=="train")))

sample_predictions$Survived=(full %>% filter(flag=="train"))$Survived

set.seed(2017)
RF_stack <- train(Survived ~ ., data = sample_predictions,method = "rf", trControl =trControl) 

set.seed(2017)
GBM_stack <- train(Survived ~ ., data = sample_predictions,method = "gbm", trControl =trControl, verbose = FALSE) 

View(full %>% filter(flag=="test"))
test_predictions <- 
  data.frame(RF_tune = predict(rf_tune,full %>% filter(flag=="test")), knn = predict(knn_model, full %>% filter(flag=="test")),
             GBM = predict(gbm_model,full %>% filter(flag=="test")), svm = predict(svm_model, full %>% filter(flag=="test")), nnet = predict(nnet_model, full %>% filter(flag=="test")))


df_final <- data.frame(PassengerId = c(892:1309), Survived=predict(RF_stack,test_predictions));
write.csv(df_final, "my_tune.csv", row.names =F, quote=F);



