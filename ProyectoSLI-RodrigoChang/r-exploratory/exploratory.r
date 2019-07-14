

titanic <- read.csv("../titanic.csv")

library(ggplot2)

# SibSp
ggplot(titanic, aes(x = SibSp, y = passenger_survived)) + 
  geom_jitter(width = 0.2, height = 0.4, alpha = 0.3)

# Parch
ggplot(titanic, aes(x = Parch, y = passenger_survived)) + 
  geom_jitter(width = 0.2, height = 0.4, alpha = 0.3)

# Fare
ggplot(titanic, aes(x = Fare, y = passenger_survived)) + 
  geom_jitter(width = 0, height = 0.4, alpha = 0.3)

# Embarked
ggplot(titanic, aes(x = Embarked, y = passenger_survived)) + 
  geom_jitter(width = 0.2, height = 0.4, alpha = 0.3)

# passenger_class
ggplot(titanic, aes(x = passenger_class, y = passenger_survived)) + 
  geom_jitter(width = 0.2, height = 0.4, alpha = 0.3)

# passenger_sex
ggplot(titanic, aes(x = passenger_sex, y = passenger_survived)) + 
  geom_jitter(width = 0.2, height = 0.4, alpha = 0.3)
