
# Importar librariea

library(tidyverse)
library(skimr)

# Leer hoja de datos

diamonds <- read_csv("diamantes.csv")

# Estadisticos descriptivos

summary(diamonds)
skim(diamonds)

# Exploracion grafica

# Variable dependiente: Price

ggplot(diamonds,aes(x=price)) +
  geom_histogram()

ggplot(diamonds,aes(x=price)) +
  geom_density()


# Graficos bivariables
# Variables Numericas (Scatter plots)

ggplot(diamonds,aes(x=carat,y=price)) +
  geom_point()

ggplot(diamonds,aes(x=depth,y=price)) +
  geom_point()

ggplot(diamonds,aes(x=table,y=price)) +
  geom_point()

ggplot(diamonds,aes(x=x,y=price)) +
  geom_point()

ggplot(diamonds,aes(x=y,y=price)) +
  geom_point()

ggplot(diamonds,aes(x=z,y=price)) +
  geom_point()


# Ahora variables categoricas

ggplot(diamonds,aes(x=cut,y=price,fill=cut)) +
  geom_boxplot()

ggplot(diamonds,aes(x=color,y=price,fill=color)) +
  geom_boxplot()

ggplot(diamonds,aes(x=clarity,y=price,fill=clarity)) +
  geom_boxplot()


# Histogramas de Precio por Cut, Color & Clarity 

ggplot(diamonds,aes(x=price,fill=cut)) + 
  geom_histogram() +
  facet_wrap(~cut)

ggplot(diamonds,aes(x=price,fill=color)) + 
  geom_histogram() +
  facet_wrap(~color)

ggplot(diamonds,aes(x=price,fill=clarity)) + 
  geom_histogram() +
  facet_wrap(~clarity)


# Precio promedio como funcion de variables categoricas (diagrama de barras)

ggplot(diamonds,aes(x=cut)) +
  geom_bar()

ggplot(diamonds,aes(x=cut,y=price,fill=cut)) +
  geom_bar(stat="summary",fun=mean)

ggplot(diamonds,aes(x=color,y=price,fill=color)) +
  geom_bar(stat="summary",fun=mean)

ggplot(diamonds,aes(x=clarity,y=price,fill=clarity)) +
  geom_bar(stat="summary",fun=mean)
