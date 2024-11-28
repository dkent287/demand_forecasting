library(fable)
library(fabletools)
library(feasts)
library(forecast)
library(tsibble)
library(dplyr)
library(lubridate)
library(fpp3)
library(ggplot2)
library(GGally)
library(latex2exp)
library(forecast)
library(Metrics)

setwd("C:/Users/dkent/OneDrive - Trimac Management Services/resource_planning/r_processes")

trimac <- read.csv('data.csv', header = TRUE)

# Setting frequency to 7 (a week)
# [['C00960','C00170','A03070','C00710','A03450','C00030']]

# Setting frequency to 7 (a week)
df <- ts(trimac$A03070, frequency = 7)

ggAcf(df, lag = 60)

df_diff <- diff(df)

ggAcf(df_diff, lag = 60)

autoplot(df_diff)

# train/ test split
test_length <- 39
train <- head(df, length(df) - test_length)
test <- tail(df, test_length)

# training an arima model
custom_model <- Arima(train,
                      order = c(1,0,3),
                      list(order=c(1,0,1),period=52), method="ML", include.mean=FALSE)

# making forecasts and plot - normal
preds_custom <- forecast(custom_model,
                         h = test_length,
                         level = c(80, 95), bootstrap = F)
autoplot(preds_custom, main='Forecast with Calculated Prediction Intervals') + 
  autolayer(test)

# making forecasts and plot - bootstrap
preds_custom <- forecast(custom_model,
                         h = test_length,
                         level = c(80, 95), bootstrap = T)
autoplot(preds_custom,main='Forecast with Bootstrapped Prediction Intervals') + 
  autolayer(test)

summary(custom_model)




