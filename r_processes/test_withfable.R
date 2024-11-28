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

setwd("C:/Users/dkent/OneDrive - Trimac Management Services/resource_planning/r_processes")

trimac <- read.csv('data.csv', header = TRUE)

# convert to tsibble
# [['C00960','C00170','A03070','C00710','A03450','C00030']]
Miles <- trimac %>%
  mutate(index = lubridate::ymd(index)) %>%
  rename(Date = index) %>%
  rename(Value = A03070) %>%
  as_tsibble(index = Date) %>%
  select(Value) %>%
  relocate(Date)

# create train test split
test_length = 39
train <- Miles %>% slice_head(n = nrow(Miles) - test_length)
test <- Miles %>% slice_tail(n = test_length)

fit <- Miles %>%
  model(
    arima001 = ARIMA(Value ~ pdq(2,1,2) + PDQ(1,0,0))
  )

fit %>% forecast(h=39) %>%
  filter(.model=='arima001') %>%
  autoplot(Miles) + 
  labs(y = "% of GDP", title = "Egyptian exports")

class(fit)


