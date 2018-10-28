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
library('gee')
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

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
#For example, Collins, Mr. John (.,) is checking for the part before the comma followed by a space . IT will select Collins, (\..) is looking for string after DOT(Including DOT) . 
#IT will select .John so you are just left with Mr
#full$Title<- sapply(full$Name,function(x) strsplit(x, split = '[,.]')[[1]][2])  
           
table(full$Sex, full$Title)

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

ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()


full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

full$Fsizemy[full$Fsize == 1] <- 'singleton'
full$Fsizemy[full$Fsize <= 4 & full$Fsize >=2] <- 'small'
full$Fsizemy[full$Fsize <= 7 & full$Fsize >=5] <- 'large'
full$Fsizemy[full$Fsize >=8] <- 'huge'

mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)

full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

full$Embarked[c(62, 830)] <- 'C'

ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)


factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf')
mice_output <- complete(mice_mod)

par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))


full$Age <- mice_output$Age


full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

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

md.pattern(full)

set.seed(754)

factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           FsizeD + Child + Mother,
                         data = train)


plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
importance    <- importance(rf_model)

varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))


ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

prediction <- predict(rf_model, test)

solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
write.csv(solution, file = 'rf_mod_Solution.csv', row.names =F)

###############################################################################

full$Tickethead=factor(substr(full$Ticket,1,1))
full$Fsizemy=factor(full$Fsizemy)

full=data.frame(full)
train <- full[which(full$flag=="train"),]
table(train$Survived)
test <- full[which(full$flag=="test"),]


check=function(x)
{
  p=predict(x,train,type="class")
  s=train$Survived
  sum(p==s)/length(s)
}

model1=randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           FsizeD + Child + Mother,
                         data = train)



model1=randomForest(factor(Survived) ~ Sex,
                    data = train)

a=function(x)
{factor(train$x))}
a(Sex)
write.csv(train,"why.csv")

model2=randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                      Fare + Embarked + Title +Tickethead+
                      Fsizemy + Child ,ntree=600,
                    data = train)

model3=rpart(factor(Survived) ~ Age + Sex + Pclass + Fsize, method="class",
             data=train)

model4=rpart(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
               Fare + Embarked + Title +Tickethead+
               Fsizemy + Child + Mother, method="class",
             data=train)


model5=rpart(factor(Survived) ~ Title + Tickethead + Fsizemy + 
               Fare + Sex + Age + Pclass, method="class",
             data=train)

model6=randomForest(factor(Survived) ~ Title + Tickethead + Fsizemy + 
                      Fare + Sex + Age + Pclass,
                    data = train)


g1=glm(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
         Fare + Embarked + Title +Tickethead+
         Fsizemy + Child + Mother,family=binomial(link='logit'),data=train)

g0=glm(factor(Survived) ~ 1,family=binomial(link='logit'),data=train)
step(g1) #backwards
step(g0,scope=list(lower=formula(g0),upper=formula(g1)), direction="forward") #forward
step(g0,scope=list(lower=formula(g0),upper=formula(g1)),direction="both",trace=0) #stepwise


g_back=glm(formula = factor(Survived) ~ Pclass + Sex + Age + Fare + 
      Title + Fsizemy, family = binomial(link = "logit"), data = train)

g_forward=glm(formula = factor(Survived) ~ Title + Tickethead + Fsizemy + 
      Fare + Sex + Age + Pclass, family = binomial(link = "logit"), 
    data = train)
g_step=glm(formula = factor(Survived) ~ Title + Fsizemy + Fare + Sex + 
      Age + Pclass, family = binomial(link = "logit"), data = train)



glm_check=function(x)
{
  p=ifelse (predict(x,train,type="response")>0.5,1,0)
  s=full$Survived[1:891]
  sum(p==s)/length(s)
}


check(model1)
check(model2)
check(model3)
check(model4)
glm_check(g_back)
glm_check(g_forward)
glm_check(g_step)

check(model5)
check(model6)







prediction <- predict(model2, test)

solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
write.csv(solution, file = 'my_v1.csv', row.names =F)








