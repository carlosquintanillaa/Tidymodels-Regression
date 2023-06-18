# Cargar libraries 

library(tidyverse)
library(skimr)

bank <- read_csv("bank.csv")
skim(bank)

# Variable dependiente : y

bank %>% 
  count(y)

ggplot(bank,aes(x=y,fill=y)) +
  geom_bar()

# Variable dependiente vs Variables Independientes (Categoricas)

# Marital

ggplot(bank,aes(x=marital,fill=y)) +
  geom_bar()

ggplot(bank,aes(x=marital,fill=y)) +
  geom_bar(position="fill")

bank %>% 
  mutate(ynum = if_else(y=="yes",1,0)) %>% 
  ggplot(aes(x=marital,y=ynum,fill=marital)) +
  geom_bar(stat="summary",fun=mean)

# Job

bank %>% 
  mutate(ynum = if_else(y=="yes",1,0)) %>% 
  ggplot(aes(x=job,y=ynum,fill=job)) +
  geom_bar(stat="summary",fun=mean)

# Education

bank %>% 
  mutate(ynum = if_else(y=="yes",1,0)) %>% 
  ggplot(aes(x=education,y=ynum,fill=education)) +
  geom_bar(stat="summary",fun=mean)

# Default

bank %>% 
  mutate(ynum = if_else(y=="yes",1,0)) %>% 
  ggplot(aes(x=default,y=ynum,fill=default)) +
  geom_bar(stat="summary",fun=mean)

# Housing 

bank %>% 
  mutate(ynum = if_else(y=="yes",1,0)) %>% 
  ggplot(aes(x=housing,y=ynum,fill=housing)) +
  geom_bar(stat="summary",fun=mean)

# Loan

bank %>% 
  mutate(ynum = if_else(y=="yes",1,0)) %>% 
  ggplot(aes(x=loan,y=ynum,fill=loan)) +
  geom_bar(stat="summary",fun=mean)

# Variable dependiente vs Variables Independientes (Continuas)

# Balance

bank %>% 
  mutate(ynum = if_else(y=="yes",1,0), balanceQ5 = as_factor(ntile(balance,5))) %>% 
  ggplot(aes(x=balanceQ5,y=ynum,fill=balanceQ5)) +
  geom_bar(stat="summary",fun=mean)

# Age

bank %>% 
  mutate(ynum = if_else(y=="yes",1,0), ageQ5 = as_factor(ntile(age,5))) %>% 
  ggplot(aes(x=ageQ5,y=ynum,fill=ageQ5)) +
  geom_bar(stat="summary",fun=mean)





