





setwd("C:/Users/dkent/OneDrive - Trimac Management Services/resource_planning/r_processes")

trimac <- read.csv('data.csv', header = TRUE)

Miles <- trimac %>%
  mutate(index = lubridate::ymd(index)) %>%
  rename(Date = index) %>%
  rename(Value = A03450) %>%
  as_tsibble(index = Date) %>%
  select(Value) %>%
  relocate(Date)

ECO <- Miles %>%
  model(
    STL(Value ~ trend(window = 7) +
          season(window = 'periodic'),
        robust = T)) %>%
  components() %>% 
  select(trend) %>% 
  mutate(trend = trend*0.25) %>%
  rename(Value = trend) %>%
  slice(11:146) %>% 
  relocate(Date,Value)

Miles <- Miles %>% 
  mutate(t2 = lag(Value, 0, order_by = Date)) %>% 
  select (t2) %>% 
  rename(Value = t2) %>% 
  slice(11:146) %>% 
  relocate(Date,Value)

Miles$ECO <- ECO$Value


Miles_pivot <- pivot_longer(Miles,2:3,names_to = 'Series',values_to = 'Value') %>%
  arrange(Series,Date)

Miles_pivot %>% 
  ggplot(aes(x=Date,y=Value,color=Series)) + 
  geom_line() + ggtitle('This is It')


fit <- Miles %>%
  model(
    mod1 = ARIMA(Value ~ pdq(d=0) + ECO),
    mod2 = ARIMA(Value ~ pdq(d=1) + xreg(ECO)),
    mod3 = ARIMA(Value ~ xreg(ECO))
  )

fit %>% 
  select(mod1) %>% 
  report()

fit %>% 
  select(mod2) %>% 
  report()

fit %>% 
  select(mod3) %>% 
  report()




