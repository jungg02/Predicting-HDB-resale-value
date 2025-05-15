library(readxl)
library(tree)
library(rpart)
library(ROCR)
library(kknn) # allows us to do KNN for regression and classification
library(e1071) #allows us to use Naive Bayes classifier

rm(list = ls())
df <- read_excel("~/Downloads/HDB_resale_prices/HDB_data_2021_sample.xlsx")

sum(is.na(df)) 
df = na.omit(df) #remove NA values

Q1 <- quantile(df$resale_price, 0.25)
Q3 <- quantile(df$resale_price, 0.75)
IQR <- Q3 - Q1

lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR


# Remove outliers
df <- df[df$resale_price >= lower_bound & df$resale_price <= upper_bound, ]

prices <- df[,"resale_price",drop=FALSE]
boxplot(prices, col = c("skyblue", "lightgreen", "lightcoral"), main = "", xlab = "", ylab = "resale_prices")


#Correlation Matrix
columns_to_include <- c('resale_price', 'floor_area_sqm', 'Remaining_lease','Dist_nearest_mall'
                        ,'Dist_nearest_waterbody', 'max_floor_lvl','postal_2digits_44','Dist_nearest_CC', 'Dist_nearest_GHawker'
                        , 'Dist_nearest_station', 'Dist_nearest_ADF', 'no_primary_schools_1km', 'Dist_CBD','mature')
data <- df[, columns_to_include, drop = FALSE]

data.cor = cor(data)
round(data.cor, digits = 4)

#UNSUPERVISED LEARNING
#PCA / PCR
columns_to_include <- c('resale_price', 'floor_area_sqm', 'Remaining_lease','Dist_nearest_mall'
                        ,'Dist_nearest_waterbody', 'max_floor_lvl','Dist_nearest_CC', 'Dist_nearest_GHawker'
                        , 'Dist_nearest_station','Dist_CBD','mature')

# Select specified columns
data1 <- df[, columns_to_include, drop = FALSE]

library(pls)
set.seed(234957)

prall = prcomp(data1, scale = TRUE)
biplot(prall)
prall.s = summary(prall)
prall.s$importance
scree = prall.s$importance[2,]
plot(scree, main = "Scree Plot", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", ylim = c(0,1), type = 'b', cex = .8)
pcr.fit=pcr(I(resale_price/1000)~floor_area_sqm+Remaining_lease+Dist_nearest_mall
                        +Dist_nearest_waterbody+max_floor_lvl+Dist_nearest_CC+Dist_nearest_GHawker
                        +Dist_nearest_station+Dist_CBD+mature,data=data1, scale=TRUE, validation="CV")

resalef = data1$resale_price[2:5865]
tsdata=cbind(resalef,data1[1:5864,])

pcr.fit=pcr(I(resale_price/1000)~floor_area_sqm+Remaining_lease+Dist_nearest_mall
            +Dist_nearest_waterbody+max_floor_lvl+Dist_nearest_CC+Dist_nearest_GHawker
            +Dist_nearest_station+Dist_CBD+mature,data=tsdata, scale=TRUE, validation="CV")
plot(pcr.fit, "loadings", comps = 1:5, legendpos = "topleft")
abline(h = 0)
validationplot(pcr.fit, val.type="MSEP", main="CV",legendpos = "topright")
pcr.pred=predict(pcr.fit, newdata=tsdata, ncomp=6)
mean((tsdata$resalef/1000-pcr.pred)^2)
#MSE = 36772.25
dev.off()


#SUPERVISED LEARNING
ntrain=2933
set.seed(567834)
tr = sample(1:nrow(df),ntrain)
train = df[tr,] 
test = df[-tr,]



#MULTIPLE LINEAR REGRESSION
lm1 = lm(I(resale_price/1000) ~ Remaining_lease  + floor_area_sqm 
         + Dist_nearest_mall + mature + max_floor_lvl + Dist_nearest_GHawker 
         + Dist_nearest_CC + Dist_nearest_station 
         + no_primary_schools_1km + Dist_CBD + Dist_nearest_ADF , data = train)
pred = predict(lm1,newdata=test)
mean((test$resale_price/1000-pred)^2) 
summary(lm1)
summary(lm1)$adj.r.squared

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm1)

# Removing additive assumption: Inclusion of quadratic terms
lm2 = lm(I(resale_price/1000) ~ Remaining_lease + I(Remaining_lease^2) + floor_area_sqm + 
         + Dist_nearest_mall + mature + max_floor_lvl + Dist_nearest_GHawker 
         + Dist_nearest_CC + Dist_nearest_station 
         + no_primary_schools_1km + Dist_CBD, data = train)
summary(lm2)
par(mfrow = c(2, 2))
plot(lm2)
pred2 = predict(lm2,newdata=test)
mean((test$resale_price/1000-pred2)^2)
#MSE = 3732.551

# Diagnostic plots
par(mfrow = c(1, 2))
plot(lm1, which = 1)
plot(lm2, which = 1)
dev.off()



#DECISION TREE
big.tree = rpart(I(resale_price/1000)~.,method="anova",data=train, minsplit=5,cp=.0005)
length(unique(big.tree$where))
plotcp(big.tree) 
bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"]
best.tree = prune(big.tree,cp=bestcp) 
#bestcp is small at 5.05*10^-4, small penalty on complexity resulting 
#in huge trees

plot(best.tree,uniform=TRUE)
text(best.tree,digits=4,use.n=TRUE,fancy=FALSE,bg='lightblue',cex=0.5)
treefit=predict(best.tree,newdata=test,type="vector") 
mean((test$resale_price/1000-treefit)^2)
#MSE=2828.732
dev.off()


#K-NEAREST NEIGHBOURS
df.loocv=train.kknn(I(resale_price/1000) ~ Remaining_lease  + floor_area_sqm 
                    + Dist_nearest_mall + Dist_nearest_waterbody + max_floor_lvl
                    + Dist_nearest_CC + Dist_nearest_station  + no_primary_schools_1km 
                    + Dist_CBD + mature,data=train,kmax=100, kernel = "rectangular")
plot((1:100),df.loocv$MEAN.SQU, type="l", col = "blue", main="LOOCV MSE", xlab="Complexity: K", ylab="MSE")
kbest=df.loocv$best.parameters$k
#kbest = 3
knnpredcv=kknn(I(resale_price/1000) ~ Remaining_lease  + floor_area_sqm 
               + Dist_nearest_mall + Dist_nearest_waterbody + max_floor_lvl
               + Dist_nearest_CC + Dist_nearest_station + mature + no_primary_schools_1km 
               + Dist_CBD,train,test,k=kbest,kernel = "rectangular")

plot(test$resale_price/1000, knnpredcv$fitted.values, main = "KNN Regression Predictions vs Actual Resale Prices", xlab = "Actual Resale Prices (in thousands)", ylab = "Predicted Resale Prices (in thousands)") 
abline(0, 1, col = "red")  # Adding a legend 
legend("topleft", legend = "Perfect Prediction Line", col = "red", lty = 1)
head(knnpredcv$fitted.values)
mean((test$resale_price/1000-knnpredcv$fitted.values)^2)
#MSE=3410.46
dev.off() 


