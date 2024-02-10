library(dplyr)
library(ggplot2)
library(GGally)

setwd("E:\\DSBA\\Unit-2\\Project file")
House = read.csv("Housing.csv")
View(House)
dim(House)

# new 4 column create according from "ocean_proximity" 
House$OceanProximity_NearBay <- ifelse(House$ocean_proximity == "NEAR BAY", 1, 0)
House$OceanProximity_Inland <- ifelse(House$ocean_proximity == "INLAND", 1, 0)
House$OceanProximity_Island <- ifelse(House$ocean_proximity == "ISLAND", 1, 0)
House$OceanProximity_NearOcean <- ifelse(House$ocean_proximity == "NEAR OCEAN", 1, 0)

dim(House)
View(House)
summary(House)

# missing Value check & remove
any(is.na(House)) # or anyNA(House)
sum(is.na(House))
House=na.omit(House)
dim(House)

House= select(House, - ocean_proximity)

#correlation check
round(cor(House, use= "pairwise.complete.obs"),2)

# Now Linear model test:
lm.Model = lm(median_house_value ~ . , data= House)
summary(lm.Model)

House= select(House, - OceanProximity_NearBay)
lm.Model2 = lm(median_house_value ~ . , data= House)
summary(lm.Model2)
anova(lm.Model2)


# Non-linearity of the relationships (but there is no Non Linear relation)
plot(lm.Model2)

# Nonlm.Model = lm(median_house_value ~ median_income + median_income^2 + households + population + total_bedrooms + housing_median_age + total_rooms , data= House)
# summary(Nonlm.Model)
# (but there is no Non Linear relation)

# plotting
plot(House$median_house_value, House$median_income, xlab = "Median House Value", ylab = "median_income")
plot(House$median_house_value, House$total_rooms, xlab = "Median House Value", ylab = "total_rooms")
plot(House$median_house_value, House$housing_median_age)
plot(House$median_house_value, House$longitude)

# Outliers Check
boxplot(House$median_house_value, ylab = "median_house_value")
boxplot(House$median_income, ylab = "median_income")
boxplot(House$total_rooms, ylab = "total_rooms")
boxplot(House$housing_median_age, ylab = "housing_median_age")
boxplot(House$total_bedrooms, ylab = "total_bedrooms")
boxplot(House$population, ylab = "population")

### Winsorizing apply for remove outliers
install.packages("DescTools")
library(DescTools)

# outliers remove in median_house_value
Q1 <- quantile(House$median_house_value, 0.25)
Q3 <- quantile(House$median_house_value, 0.75)
IQR <- Q3 - Q1
House$median_house_value_Win<- Winsorize(House$median_house_value, minval=Q1-1.5*IQR, maxval=Q3+1.5*IQR)
boxplot(House$median_house_value_Win, ylab = "median_house_value_Win")

# outliers remove in median_income
Q1 <- quantile(House$median_income, 0.25)
Q3 <- quantile(House$median_income, 0.75)
IQR <- Q3 - Q1
House$median_income_Win <- Winsorize(House$median_income, minval=Q1-1.5*IQR, maxval=Q3+1.5*IQR)
boxplot(House$median_income_Win, ylab = "median_income_Win")

# outliers remove in total_rooms
Q1 <- quantile(House$total_rooms, 0.25)
Q3 <- quantile(House$total_rooms, 0.75)
IQR <- Q3 - Q1
House$total_rooms_Win <- Winsorize(House$total_rooms, minval=Q1-1.5*IQR, maxval=Q3+1.5*IQR)
boxplot(House$total_rooms_Win, ylab = "total_rooms_Win")

# outliers remove in total_bedrooms
Q1 <- quantile(House$total_bedrooms, 0.25)
Q3 <- quantile(House$total_bedrooms, 0.75)
IQR <- Q3 - Q1
House$total_bedrooms_Win <- Winsorize(House$total_bedrooms, minval=Q1-1.5*IQR, maxval=Q3+1.5*IQR)
boxplot(House$total_bedrooms_Win, ylab = "total_bedrooms_Win")

# outliers remove in population
Q1 <- quantile(House$population, 0.25)
Q3 <- quantile(House$population, 0.75)
IQR <- Q3 - Q1
House$population_Win<- Winsorize(House$population, minval=Q1-1.5*IQR, maxval=Q3+1.5*IQR)
boxplot(House$population_Win, ylab = "population_Win")


View(House)
dim(House)
House= select(House, -c(median_house_value,median_income,total_rooms,total_bedrooms,population,households,OceanProximity_NearOcean))
dim(House)

#  Now again Checking the lm Model without Outliers value
lm.Model4 = lm(median_house_value_Win ~ . , data= House)
summary(lm.Model4)
anova(lm.Model4)
plot(lm.Model4)


House$prediction <- predict(lm.Model4)
House$residual <- resid(lm.Model4)
View(House)
head(House)
head(House,10)

plot(House$prediction, House$residual, main = "Residuals vs. predicted price", xlab = "Predicted Price", ylab = "Residuals")

#### The End ####










