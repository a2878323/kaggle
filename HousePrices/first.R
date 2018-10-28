#資料視覺化
library(ggplot2)
library(ggthemes)
library(scales)
library(gridExtra)
#資料整理
library(dplyr) # data manipulation
#補遺失值
library(mice) # imputation
#機器學習模型
library(randomForest) # 隨機森林
library(rpart) #決策樹
library(nnet) #類神經網路
library(neuralnet) #類神經網路
library(e1071) #SVM
library(gbm) #GBM
library(kernlab) #SVM
library(klaR)
library(VGAM)
#模型訓練
library(caret) 
library(class)
library(caTools)

#讀取資料
rm(list = ls())
setwd("~/Documents/Kaggle/HousePrices")
train=read.csv("train.csv",header=T,stringsAsFactors = F)
test=read.csv("test.csv",header=T,stringsAsFactors = F)
train$flag="train"
test$flag="test"
all=bind_rows(train,test)

summary(all)

# training control for caret 不懂，不懂參數要怎麼放
trControl <- trainControl(method="repeatedcv", number=7, repeats=5);

#遺失值
#LotFrontage MasVnrArea BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF BsmtFullBath BsmtHalfBath
#GarageYrBlt GarageCars GarageArea
hist(all$SalePrice)

#取出欄位名
col=names(all)[1:81]

#單因子迴歸檢視變數是否重要
p=c();e=c()
for(x in 1:length(col)) 
{
  e[x]=summary(lm(SalePrice~train[,x],data=train))$coefficients[2,1]
  p[x]=summary(lm(SalePrice~train[,x],data=train))$coefficients[2,4]
  print(x)
}
#取出p value<0.2之變數與其係數
p_2=which(p<=0.2)
View(cbind(col,e,p)[p_2,])
#只留這些因子變數
interest_par=all[,c(1,p_2,which(names(all)=="MoSold"),which(names(all)=="YrSold"),which(names(all)=="flag"))]
interest_par$flag=all$flag
interest_par$PoolQC=all$PoolQC

summary(interest_par)

table(MasVnrArea)
names(interst_par)

interest_par$tarea=interest_par$YrSold-interest_par$YearRemodAdd

#補文字型遺失值
char_mis=c(
'MSZoning','Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
'BsmtFinType2','KitchenQual','Functional','FireplaceQu','GarageType',
'GarageFinish','GarageQual','Fence','MiscFeature','SaleType','PoolQC')


interest_par[char_mis] <- lapply(interest_par[char_mis], function(x) ifelse(is.na(x),"Missing",x))

#文字型變數轉類別
factor_vars <- c('MSZoning','Street','Alley','LotShape','LandContour','LotConfig',
                 'LandSlope','Neighborhood','BldgType','HouseStyle','MasVnrType',
                 'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
                 'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
                 'CentralAir','KitchenQual','Functional','FireplaceQu','GarageType',
                 'GarageFinish','GarageQual','Fence','MiscFeature','SaleType','PoolQC')

interest_par[factor_vars] <- lapply(interest_par[factor_vars], function(x) as.factor(x))


#補MasVnrArea
na1=train(MasVnrArea~
            MSSubClass+MSZoning+LotFrontage+LotArea+  
            Street+Alley+LotShape+LandContour+LotConfig+
            LandSlope+ Neighborhood +BldgType+  HouseStyle+OverallQual+
            OverallCond+YearBuilt+ YearRemodAdd+MasVnrType+ 
            ExterQual+ ExterCond+ Foundation+BsmtQual+  BsmtCond+ 
            BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+BsmtUnfSF+
            TotalBsmtSF+Heating+   HeatingQC+ CentralAir+X1stFlrSF+
            X2ndFlrSF+ GrLivArea+ BsmtFullBath+FullBath+  HalfBath+ 
            BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+Functional+
            Fireplaces+FireplaceQu+GarageType+GarageYrBlt+GarageFinish+
            GarageCars+GarageArea+GarageQual+WoodDeckSF+OpenPorchSF+
            EnclosedPorch+X3SsnPorch+ScreenPorch+PoolArea+  Fence+    
            MiscFeature +MoSold+SaleType
          , data = interest_par, trControl = trControl, method="rpart", na.action = na.omit);

miss=which(is.na(interest_par$MasVnrArea))
interest_par$MasVnrArea[miss] = predict(na1, interest_par[miss,],na.action=na.pass)


#補LotFrontage
na1=train(LotFrontage~
            MSSubClass+MSZoning+LotArea+  
            Street+Alley+LotShape+LandContour+LotConfig+
            LandSlope+ Neighborhood +BldgType+  HouseStyle+OverallQual+
            OverallCond+YearBuilt+ YearRemodAdd+MasVnrType+ MasVnrArea+
            ExterQual+ ExterCond+ Foundation+BsmtQual+  BsmtCond+ 
            BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+BsmtUnfSF+
            TotalBsmtSF+Heating+   HeatingQC+ CentralAir+X1stFlrSF+
            X2ndFlrSF+ GrLivArea+ BsmtFullBath+FullBath+  HalfBath+ 
            BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+Functional+
            Fireplaces+FireplaceQu+GarageType+GarageYrBlt+GarageFinish+
            GarageCars+GarageArea+GarageQual+WoodDeckSF+OpenPorchSF+
            EnclosedPorch+X3SsnPorch+ScreenPorch+PoolArea+  Fence+    
            MiscFeature +MoSold+SaleType
          , data = interest_par, trControl = trControl, method="rpart", na.action = na.omit);
miss=which(is.na(interest_par$LotFrontage))
interest_par$LotFrontage[miss] = predict(na1, interest_par[miss,],na.action=na.pass)

#補BsmtFinSF1 BsmtUnfSF TotalBsmtSF BsmtFinSF1
which(is.na(interest_par$BsmtFinSF1))==which(is.na(interest_par$BsmtUnfSF))
which(is.na(interest_par$BsmtFinSF1))==which(is.na(interest_par$TotalBsmtSF))
miss=which(is.na(interest_par$BsmtFinSF1))
interest_par$BsmtFinSF1[miss]=0
interest_par$BsmtUnfSF[miss]=0
interest_par$TotalBsmtSF[miss]=0

plot(~BsmtExposure+BsmtFullBath)
interest_par$BsmtFullBath[which(is.na(interest_par$BsmtFullBath))]=0

#補GarageYrBlt GarageCars GarageArea"
#GarageCars
plot(~GarageType+GarageArea)
interest_par$GarageCars[which(is.na(interest_par$GarageCars))]=2
#GarageArea
d1=interest_par %>% filter(GarageType=="Detchd")
interest_par$GarageArea[which(is.na(interest_par$GarageArea))]=mean(d1$GarageArea,na.rm=T)
#GarageYrBlt
na1=train(GarageYrBlt ~ MSSubClass+MSZoning+LotFrontage+LotArea+  
            Street+Alley+LotShape+LandContour+LotConfig+
            LandSlope+ Neighborhood +BldgType+  HouseStyle+OverallQual+
            OverallCond+YearBuilt+ YearRemodAdd+MasVnrType+MasVnrArea+
            ExterQual+ ExterCond+ Foundation+BsmtQual+BsmtCond+
            BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+BsmtUnfSF+
            TotalBsmtSF+Heating+   HeatingQC+ CentralAir+X1stFlrSF+
            X2ndFlrSF+ GrLivArea+BsmtFullBath+FullBath+  HalfBath+ 
            BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+Functional+
            Fireplaces+FireplaceQu+GarageType+GarageFinish+
            GarageCars+GarageArea+GarageQual+WoodDeckSF+OpenPorchSF+
            EnclosedPorch+X3SsnPorch+ScreenPorch+PoolArea+  Fence+    
            MiscFeature +MoSold+SaleType
          , data = interest_par , trControl = trControl, method="rpart", na.action = na.omit);
miss=which(is.na(interest_par$GarageYrBlt))
interest_par$GarageYrBlt[miss] = predict(na1, interest_par[miss,],na.action=na.pass)

#房子蓋好後多久售出
interest_par$yeardiff=interest_par$YrSold-interest_par$YearBuilt
#房子蓋好後多久裝潢
interest_par$yeardiffremod=interest_par$YearRemodAdd-interest_par$YearBuilt
#裝潢後多久售出
interest_par$yearremoddiff=interest_par$YrSold-interest_par$YearRemodAdd
#售出年淡旺季
interest_par$soldtime=factor(paste0(interest_par$YrSold,ifelse(interest_par$MoSold>=4 & interest_par$MoSold<=8,"HOT","COLD")))


#總面積
attach(all)
GarageArea=ifelse(is.na(GarageArea),0,GarageArea)
MasVnrArea=ifelse(is.na(MasVnrArea),0,MasVnrArea)
TotalBsmtSF=ifelse(is.na(TotalBsmtSF),0,TotalBsmtSF)
allarea=LotArea+WoodDeckSF+OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+PoolArea+GarageArea+GrLivArea+MasVnrArea+TotalBsmtSF
interest_par$allarea=allarea
detach(all)

#計算品質次數 Ex>Gd>TA>Fa>Po>Missing
#製作dummy variable
dummy=dummyVars( ~ HeatingQC+KitchenQual+FireplaceQu+GarageQual+PoolQC+ExterQual+BsmtQual,data = interest_par)
dummy_data=predict(dummy,interest_par)
str(dummy_data)
dummy_data=as.data.frame(dummy_data)

# 加總
interest_par$Excount=dummy_data$HeatingQC.Ex+dummy_data$KitchenQual.Ex+dummy_data$FireplaceQu.Ex+dummy_data$GarageQual.Ex+dummy_data$PoolQC.Ex+dummy_data$ExterQual.Ex+dummy_data$BsmtQual.Ex
interest_par$Gdcount=dummy_data$HeatingQC.Gd+dummy_data$KitchenQual.Gd+dummy_data$FireplaceQu.Gd+dummy_data$GarageQual.Gd+dummy_data$PoolQC.Gd+dummy_data$ExterQual.Gd+dummy_data$BsmtQual.Gd
interest_par$TAcount=dummy_data$HeatingQC.TA+dummy_data$KitchenQual.TA+dummy_data$FireplaceQu.TA+dummy_data$GarageQual.TA+                    +dummy_data$ExterQual.TA+dummy_data$BsmtQual.TA
interest_par$Facount=dummy_data$HeatingQC.Fa+dummy_data$KitchenQual.Fa+dummy_data$FireplaceQu.Fa+dummy_data$GarageQual.Fa+dummy_data$PoolQC.Fa+dummy_data$ExterQual.Fa+dummy_data$BsmtQual.Fa
interest_par$Pocount=dummy_data$HeatingQC.Po+                         +dummy_data$FireplaceQu.Po+dummy_data$GarageQual.Po                                            
interest_par$Missingcount=dummy_data$KitchenQual.Missing+dummy_data$FireplaceQu.Missing+dummy_data$GarageQual.Missing+dummy_data$PoolQC.Missing+dummy_data$BsmtQual.Missing

View(interest_par[c('HeatingQC','KitchenQual','FireplaceQu','GarageQual','PoolQC','ExterQual','BsmtQual','Excount','Gdcount','TAcount','Facount','Pocount','Missingcount')])

#品質分數
interest_par$score=(interest_par$Excount*6+interest_par$Gdcount*5+interest_par$TAcount*4+interest_par$Facount*3+interest_par$Pocount*2+interest_par$Missingcount*1)

#SaleType_flag
interest_par$SaleType_flag[which(interest_par$SaleType %in% c('WD')=='TRUE')]='WD'
interest_par$SaleType_flag[which(interest_par$SaleType %in% c('New','Con')=='TRUE')]='New'
interest_par$SaleType_flag[which(interest_par$SaleType %in% c('COD','ConLD','ConLI','ConLw','CWD','Oth')=='TRUE')]='COD'
#interest_par$SaleType_flag[which(interest_par$SaleType %in% c('Oth')=='TRUE')]='Oth'
#interest_par$SaleType_flag[which(interest_par$SaleType %in% c('CWD')=='TRUE')]='CWD'
interest_par$SaleType_flag=factor(interest_par$SaleType_flag)

summary(interest_par)

model1=lm(SalePrice ~ ., data=dummy_data[1:1460,])
test=predict(model1,dummy_data[1461:2919,])



step(model1) #backwards
back=lm(formula = SalePrice ~ MSSubClass + `MSZoning.C (all)` + MSZoning.FV + 
          LotFrontage + LotArea + Street.Grvl + LotShape.IR3 + LandContour.Bnk + 
          LotConfig.CulDSac + LotConfig.FR2 + LandSlope.Gtl + LandSlope.Mod + 
          Neighborhood.BrkSide + Neighborhood.ClearCr + Neighborhood.CollgCr + 
          Neighborhood.Crawfor + Neighborhood.Edwards + Neighborhood.Gilbert + 
          Neighborhood.IDOTRR + Neighborhood.Mitchel + Neighborhood.NAmes + 
          Neighborhood.NoRidge + Neighborhood.NridgHt + Neighborhood.NWAmes + 
          Neighborhood.OldTown + Neighborhood.Sawyer + Neighborhood.SawyerW + 
          Neighborhood.StoneBr + Neighborhood.SWISU + Neighborhood.Timber + 
          BldgType.1Fam + BldgType.2fmCon + BldgType.Duplex + HouseStyle.1.5Fin + 
          HouseStyle.2.5Fin + HouseStyle.2Story + OverallQual + OverallCond + 
          YearBuilt + MasVnrType.BrkCmn + MasVnrType.BrkFace + MasVnrArea + 
          ExterQual.Ex + Foundation.CBlock + Foundation.PConc + BsmtQual.Ex + 
          BsmtExposure.Av + BsmtExposure.Gd + BsmtFinType1.ALQ + BsmtFinType1.BLQ + 
          BsmtFinType1.GLQ + BsmtFinType1.Rec + Heating.OthW + HeatingQC.Ex + 
          X2ndFlrSF + GrLivArea + BsmtFullBath + FullBath + HalfBath + 
          BedroomAbvGr + KitchenAbvGr + KitchenQual.Ex + TotRmsAbvGrd + 
          Functional.Maj2 + Functional.Min1 + Functional.Min2 + Functional.Sev + 
          Fireplaces + FireplaceQu.Ex + GarageType.2Types + GarageType.Attchd + 
          GarageType.BuiltIn + GarageType.Detchd + GarageCars + GarageArea + 
          GarageQual.Ex + GarageQual.Fa + WoodDeckSF + X3SsnPorch + 
          ScreenPorch + SaleType.Con + SaleType.New, data = dummy_data[1:1460,]) 
                                                                       

set.seed(107)
dt=train(SalePrice ~MSSubClass+MSZoning+LotFrontage+LotArea+  
           Street+Alley+LotShape+LandContour+LotConfig+
           LandSlope+ Neighborhood +BldgType+  HouseStyle+OverallQual+
           OverallCond+YearBuilt+MasVnrType+ MasVnrArea+
           ExterQual+ Foundation+BsmtQual+
           BsmtExposure+BsmtFinType1+Heating+HeatingQC+
           X2ndFlrSF+ GrLivArea+ BsmtFullBath+FullBath+  HalfBath+ 
           BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+Functional+
           Fireplaces+FireplaceQu+GarageType+
           GarageCars+GarageArea+GarageQual+WoodDeckSF+
           X3SsnPorch+ScreenPorch+SaleType
          , data = interest_par[1:1460,], trControl = trControl, method="rpart");

set.seed(107)
rf=train(log(SalePrice) ~LotFrontage+LotArea+
           Neighborhood +OverallQual+
           OverallCond+YearBuilt+ MasVnrArea+
           ExterQual+BsmtQual+
           BsmtExposure+BsmtFinType1+HeatingQC+
           X2ndFlrSF+ GrLivArea+FullBath+   
           BedroomAbvGr+KitchenQual+TotRmsAbvGrd+
           Fireplaces+FireplaceQu+GarageType+
           GarageCars+GarageArea+WoodDeckSF+
           yeardiff+yearremoddiff+allarea+Excount+Gdcount+score+Saletype_flag
         , data = interest_par[1:1460,], trControl = trControl, method="rf");



set.seed(107)
rf2=randomForest(log(SalePrice) ~LotFrontage+LotArea+
                    Neighborhood +OverallQual+
                   OverallCond+YearBuilt+ MasVnrArea+
                   ExterQual+BsmtQual+
                   BsmtExposure+BsmtFinType1+HeatingQC+
                   X2ndFlrSF+ GrLivArea+FullBath+   
                   BedroomAbvGr+KitchenQual+TotRmsAbvGrd+
                   Fireplaces+FireplaceQu+GarageType+
                   GarageCars+GarageArea+WoodDeckSF+
                   yeardiff+yearremoddiff+allarea+Excount+Gdcount+score+SaleType_flag,
                   data=interest_par[1:1460,])


order(importance(rf2))

importance(rf2)[ order(importance(rf2)), ]

er=(exp(predict(rf2,interest_par[1:1460,]))-interest_par$SalePrice[1:1460])
which(error>10000)
View(interest_par[which(error>10000)])

check=function(x)
{
  sqrt(mean(((exp(predict(x,interest_par[1:1460,]))-interest_par$SalePrice[1:1460]))^2))
}
check(rf)
check(rf2)
check(rf3)

check=function(x)
{
  sqrt(mean((((predict(x,interest_par[1:1460,]))-interest_par$SalePrice[1:1460]))^2))
}

check(dt)
check(back)


submit <- data.frame(Id = c(1461:2919), SalePrice=exp(predict(rf2,interest_par[1461:2919,])))
write.csv(submit, "v1_3.csv", row.names =F, quote=F);




MSSubClass+MSZoning+LotFrontage+LotArea+  
  Street+Alley+LotShape+LandContour+LotConfig+
  LandSlope+ Neighborhood +BldgType+  HouseStyle+OverallQual+
  OverallCond+YearBuilt+ YearRemodAdd+MasVnrType+ MasVnrArea+
  ExterQual+ ExterCond+ Foundation+BsmtQual+  BsmtCond+ 
  BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+BsmtUnfSF+
  TotalBsmtSF+Heating+   HeatingQC+ CentralAir+X1stFlrSF+
  X2ndFlrSF+ GrLivArea+ BsmtFullBath+FullBath+  HalfBath+ 
  BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+Functional+
  Fireplaces+FireplaceQu+GarageType+GarageYrBlt+GarageFinish+
  GarageCars+GarageArea+GarageQual+WoodDeckSF+OpenPorchSF+
  EnclosedPorch+X3SsnPorch+ScreenPorch+PoolArea+  Fence+    
  MiscFeature +MoSold+SaleType

