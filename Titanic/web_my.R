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
######################################################
rm(list = ls())
setwd("~/Documents/Kaggle/Titanic")
train=read.csv("train.csv",header=T,stringsAsFactors = F)
test=read.csv("test.csv",header=T,stringsAsFactors = F)
train$flag="train"
test$flag="test"
full=bind_rows(train,test)
str(full)
summary(full)
md.pattern(full)

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
#For example, Collins, Mr. John (.,) is checking for the part before the comma followed by a space . IT will select Collins, (\..) is looking for string after DOT(Including DOT) . 
#IT will select .John so you are just left with Mr
#full$Title<- sapply(full$Name,function(x) strsplit(x, split = '[,.]')[[1]][2])  

#table(full$Sex, full$Title)

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'
table(full$Sex, full$Title)
full$Surname <- sapply(full$Name,function(x) strsplit(x, split = '[,.]')[[1]][1])

full$Fsize <- full$SibSp + full$Parch + 1
full$Family <- paste(full$Surname, full$Fsize, sep='_')

full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

full$Fsizemy[full$Fsize == 1] <- 'singleton'
full$Fsizemy[full$Fsize <= 4 & full$Fsize >=2] <- 'small'
full$Fsizemy[full$Fsize <= 7 & full$Fsize >=5] <- 'large'
full$Fsizemy[full$Fsize >=8] <- 'huge'

full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)

full$Embarked[c(62, 830)] <- 'C'
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

 

full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)


set.seed(123)
random=sample(length(train[,1]),length(train[,1])*0.7)

train_test=data[-random,]
train_train=data[random,]
test$flag="test"
train$flag="train"




full$FamilyN=sapply(full$Name,function(x) strsplit(x,"[,.]")[[1]][1])
p1=data.frame(table(full$FamilyN))
colnames(p1)[colnames(p1) == 'Var1'] <- 'FamilyN'
colnames(p1)[colnames(p1) == 'Freq'] <- 'FamilyNumber'
full=merge(full,p1,by="FamilyN")
full$Familyflag=sapply(full$FamilyNumber ,function(x) ifelse(x>1,1,0))

p2=data.frame(table(full$Ticket))
colnames(p2)[colnames(p2) == 'Var1'] <- 'Ticket'
colnames(p2)[colnames(p2) == 'Freq'] <- 'TicketNumber'
full=merge(full,p2,by="Ticket")
full$Ticketflag=sapply(full$TicketNumber ,function(x) ifelse(x>1,1,0))

full$groupflag=ifelse((full$Ticketflag+full$Familyflag>0),1,0)
full$Tickethead=substr(full$Ticket,1,1)



set.seed(754)

factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD',
                 'Fsizemy','Tickethead','groupflag','Ticketflag','Familyflag','FamilyN')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

md.pattern(full)



###############################################################################

fullmodel=factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
  Fare + Embarked + Title + 
  FsizeD + Child + Mother+Fsizemy+Tickethead+Ticketflag+TicketNumber+groupflag+Familyflag+FamilyNumber


train <- full[which(full$flag=="train"),]
table(train$Survived)
test <- full[which(full$flag=="test"),]
n=nrow(train)

check=function(x)
{
  p=predict(x,train,type="class")
  s=train$Survived
  sum(p==s)/length(s)
}



bpn <- neuralnet(formula = Survived ~  Age + SibSp + Parch + 
                   Fare 
              
                 , 
                 data = train,
                 hidden = c(2),       # 一個隱藏層：2個node
                 learningrate = 0.01, # learning rate
                 threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
                 stepmax = 5e5        # 最大的ieration數 = 500000(5*10^5)
                 
)

kv <- round(sqrt(n))

train4 <- names(train3) %in% c('Survived','Age','SibSp','Parch','Fare'
                               )

,'FsizeD','Child','Mother','Fsizemy','Tickethead',
                               'Ticketflag','TicketNumber','groupflag','Familyflag',
                               'FamilyNumber')
train4=train3[train4]                                
                               
train2=train
train2$Cabin = NULL
train2$Deck = NULL
train2$Ticket = NULL
train2$Name = NULL
train2$flag = NULL
train3=train2
train2=train3[,1:5]
kmean <- knn(train4, train4, factor(train4$Survived), 5)

md.pattern(train2)


model1=randomForest(fullmodel,
                    data = train)
importance(model1)


model2=randomForest(factor(Survived)~Pclass + Sex + Age + SibSp  + 
                      Fare  + Title + 
                      FsizeD  +Tickethead+TicketNumber+FamilyNumber
                    ,
                    data = train)
importance(model2)



model3=rpart(fullmodel, method="class",
             data=train)

model4=rpart(factor(Survived) ~ Age + Sex + Pclass + Fsize, method="class",
             data=train)

model5=rpart(factor(Survived) ~Pclass + Sex + Age + SibSp + Parch + 
               Fare + Embarked + Title + 
               FsizeD + Child +Fsizemy+Tickethead+Ticketflag+TicketNumber+groupflag+FamilyNumber, method="class",
             data=train)


model6=rpart(factor(Survived) ~ Title + Tickethead + Fsizemy + 
               Fare + Sex + Age + Pclass, method="class",
             data=train)



g1=glm(fullmodel,family=binomial(link='logit'),data=train)

g0=glm(factor(Survived) ~ 1,family=binomial(link='logit'),data=train)
step(g1) #backwards
step(g0,scope=list(lower=formula(g0),upper=formula(g1)), direction="forward") #forward
step(g0,scope=list(lower=formula(g0),upper=formula(g1)),direction="both",trace=0) #stepwise


g_back=glm(formula = factor(Survived) ~ Pclass + Sex + Age + Fare + 
             Title + Fsizemy, family = binomial(link = "logit"), data = train)

g_forward=glm(formula = factor(Survived) ~Title + Tickethead + Fsizemy + 
                Fare + Sex + groupflag, family = binomial(link = "logit"), 
              data = train)
g_step=glm(formula = factor(Survived) ~Title + Tickethead + Fsizemy + 
             Fare + Sex + groupflag, family = binomial(link = "logit"), data = train)



glm_check=function(x)
{
  p=ifelse (predict(x,train,type="response")>0.5,1,0)
  s=train$Survived
  sum(p==s)/length(s)
}


check(model1)
check(model2)
check(model3)
check(model4)
check(model5)
check(model6)
glm_check(g_back)
glm_check(g_forward)
glm_check(g_step)

check(model5)
check(model6)







prediction <- predict(model2, test)

