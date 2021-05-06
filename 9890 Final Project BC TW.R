library(tidyverse)
library(randomForest)
library(glmnet)
library(tictoc)


# prep stuff
# this is in a brackets just so that I don't need to run it line by line
{
  car6 <- as_tibble(read.csv('Cars6.csv'))
  car6$state_id <- as.factor(car6$state_id)
  car6$car_origin <- as.factor(car6$car_origin)
  car6$engine_cylinders <- as.factor(car6$engine_cylinders)
  car6$engine_type <- as.factor(car6$engine_type)
  car6$exterior_color <- as.factor(car6$exterior_color)
  car6$fleet <- as.factor(car6$fleet)
  car6$franchise_dealer <- as.factor(car6$franchise_dealer)
  car6$franchise_make <- as.factor(car6$franchise_make)
  car6$fuel_type <- as.factor(car6$fuel_type)
  car6$has_accidents <- as.factor(car6$has_accidents)
  car6$interior_color <- as.factor(car6$interior_color)
  car6$isCab <- as.factor(car6$isCab)
  car6$is_new <- as.factor(car6$is_new)
  car6$listing_color <- as.factor(car6$listing_color)
  car6$make_name <- as.factor(car6$make_name)
  car6$model_name <- as.factor(car6$model_name)
  car6$salvage <- as.factor(car6$salvage)
  car6$transmission <- as.factor(car6$transmission)
  car6$wheel_system <- as.factor(car6$wheel_system)
  n1 <- dim(car6)[1]
  comp_time <- matrix(1:400, nrow = 100)  # store time
  v.err <- matrix(1:400, nrow = 100)  # store test residuals
  t.err <- matrix(1:400, nrow = 100)  # store train residuals
}

# Actual work on carS starts here:
{
  set.seed(317)
  carS <- car6[sample(1:n1,8000),]
  # carS %>% group_by(model_name) %>%  summarise(Count = n()) %>%
  #   arrange(desc(Count)) %>%  ungroup() %>%  print(n = Inf)
  
  # I take out a few rows because these only appear a handful of times, but they
  # would still add more factors and may be influenced by high variance
  # I left in out liers because all of these models should mitigate the influence
  carS <- carS %>% filter(
    !(franchise_make %in% c('Alfa Romeo','Shelby','Rolls-Royce','Aston Martin',
                            'Bentley','Porsche','Merecedes-Benz','Jaguar','FIAT',
                            'Chrysler','Volvo','MINI','Lexus','Maserati','Land Rover',
                            'Genesis','Audi')),
    !(make_name %in% c('Hummer','Suzuki','Isuzu','Mazda','Mitsubishi','Cadillac',
                       'Lincoln','Subaru')),
    !(model_name %in% c('Silverado 2500','Sierra 2500HD','Silverado Classic 1500HD',
                        'i-Series','B-Series','Silverado 2500HD','F-150 Heritage',
                        'Sonoma','Silverado SS','Blackwood','Equator','Silverado 1500HD',
                        'Sierra Classic 1500','Baja','Raider','S-10',
                        'Sierra 1500 Limited','Escalade EXT','H3T','Mark LT',
                        'Sierra Classic 1500HD','Silverado Classic 1500')),
    !(state_id %in% c('KS','OK','AZ','CO','HI','NV','CA'))
  )
  n <- dim(carS)[1]
  p <- dim(carS)[2]-1
  K <- 100
  test_or_train <- ifelse(1:n <= length(price.tr), 'Train', 'Test')
  
  # Trial 1:
  
  set.seed(100)
  spliter <- sample(1:n, n/5)
  train <- carS[-spliter,]
  test <- carS[spliter,]
  
  price.tr <- train$price
  price.val <- test$price
  
  # in general, t in a variable name means training,
  # v stands for validate, but I use it to mean test
  m.tr <- model.matrix(price ~ ., data = train)[,-1]  
  m.val <- model.matrix(price ~ ., data = test)[,-1]
  par(mfrow=c(1,1))
}

# Regression
# Lasso
{
tic('Lasso')
tic('Lasso CV Creation')
mod.las <- cv.glmnet(m.tr,price.tr, alpha=1) 
q <- toc()
plot(mod.las, main = 'Lasso', sub = paste ('Seconds to compute: ',q$toc-q$tic))
lambda <- mod.las$lambda.min
beta.las <- as.matrix(predict(mod.las, s = lambda, type = 'coefficients'))[-1]
pred.las <- predict(mod.las, s = lambda, newx = m.val)
tr.las <- predict(mod.las, s = lambda, newx = m.tr)
v.err[1,1] <- 1 - mean((pred.las - price.val)^2) / mean((mean(pred.las)-price.val)^2)
t.err[1,1] <- 1 - mean((tr.las - price.tr)^2) / mean((mean(tr.las)-price.tr)^2)
toc()
las.v.resid <- pred.las-price.val
las.t.resid <- tr.las-price.tr
}

# Ridge
{
tic('Ridge')
tic('Ridge CV creation')
mod.rid <- cv.glmnet(m.tr,price.tr, alpha=0) 
q <- toc()
plot(mod.rid, main = 'Ridge', sub = paste ('Seconds to compute: ',q$toc-q$tic))
lambda <- mod.rid$lambda.min
beta.rid <- as.matrix(predict(mod.rid, s = lambda, type = 'coefficients'))[-1]
pred.rid <- predict(mod.rid, s = lambda, newx = m.val)
tr.rid <- predict(mod.rid, s = lambda, newx = m.tr)
v.err[1,2] <- 1- mean((pred.rid- price.val)^2) / mean((mean(pred.rid)-price.val)^2)
t.err[1,2] <- 1- mean((tr.rid - price.tr)^2) / mean((mean(tr.rid)-price.tr)^2)
toc()
rid.v.resid <- pred.rid-price.val
rid.t.resid <- tr.rid-price.tr
}

# Elastic
{
tic('Elastic')
tic('Elastic CV creation')
mod.els <- cv.glmnet(m.tr,price.tr, alpha=0.5) 
q <- toc()
plot(mod.els, main = 'Elastic', sub = paste ('Seconds to compute: ',q$toc-q$tic))
lambda <- mod.els$lambda.min
beta.els <- as.matrix(predict(mod.els, s = lambda, type = 'coefficients'))[-1]
pred.els <- predict(mod.els, s = lambda, newx = m.val)
tr.els <- predict(mod.els, s = lambda, newx = m.tr)
v.err[1,3] <- 1 - mean((pred.els- price.val)^2) / mean((mean(pred.els)-price.val)^2)
t.err[1,3] <- 1 - mean((tr.els - price.tr)^2) / mean((mean(tr.els)-price.tr)^2)
toc()
els.v.resid <- pred.els-price.val
els.t.resid <- tr.els-price.tr
}


# Random Forest
{
tic('Random Forest')
rf.car <- randomForest(x=m.tr, y=price.tr,mtry=floor(sqrt(p)), importance=T)
y.hat.test <- predict(rf.car, newdata = m.val)
y.hat.train <- predict(rf.car, newdata = m.tr)
v.err[1,4] <- 1 - mean((y.hat.test-price.val)^2) / mean((mean(y.hat.test)-price.val)^2)
t.err[1,4] <- 1 - mean((y.hat.train-price.tr)^2) / mean((mean(y.hat.train)-price.tr)^2)
rf.import <- importance(rf.car)
toc()
rf.v.resid <- y.hat.test-price.val
rf.t.resid <- y.hat.train-price.tr
}

# bar plot for coefficients (unordered)
bar_label <- tibble('Coefficient' = colnames(m.tr), 'group' = 1:236)
for (i in 1:236){
  if (grepl('back_legroom',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'back_legroom'}
  else if (grepl('engine_displacement',bar_label$Coefficient[i])){
    bar_label$group[i] = 'engine_displacement'}
  else if (grepl('franchise_make',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'franchise_make'}
  else if (grepl('height',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'height'}
  else if (grepl('is_new',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'is_new'}
  else if (grepl('maximum_seating',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'maximum_seating'}
  else if (grepl('seller_rating',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'seller_rating'}
  else if (grepl('wheelbase',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'wheelbase'}
  else if (grepl('bed_length',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'bed_length'}
  else if (grepl('engine_type',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'engine_type'}
  else if (grepl('front_legroom',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'front_legroom'}
  else if (grepl('highway_fuel_',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'highway_fuel_'}
  else if (grepl('length',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'length'}
  else if (grepl('mileage',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'mileage'}
  else if (grepl('sp_id',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'sp_id'}
  else if (grepl('width',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'width'}
  else if (grepl('city_fuel_',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'city_fuel_'}
  else if (grepl('exterior_color',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'exterior_color'}
  else if (grepl('fuel_tank',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'fuel_tank'}
  else if (grepl('horsepower',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'horsepower'}
  else if (grepl('listed_month',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'listed_month'}
  else if (grepl('model_name',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'model_name'}
  else if (grepl('torque',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'torque'}
  else if (grepl('year',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'year'}
  else if (grepl('daysonmarket',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'daysonmarket'}
  else if (grepl('fleet',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'fleet'}
  else if (grepl('fuel_type',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'fuel_type'}
  else if (grepl('interior_color',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'interior_color'}
  else if (grepl('listing_color',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'listing_color'}
  else if (grepl('owner_count',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'owner_count'}
  else if (grepl('transmission',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'transmission'}
  else if (grepl('state_id',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'state_id'}
  else if (grepl('engine_cylinders',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'engine_cylinders'}
  else if (grepl('franchise_dealer',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'franchise_dealer'}
  else if (grepl('has_accidents',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'has_accidents'}
  else if (grepl('isCab',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'isCab'}
  else if (grepl('make_name',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'make_name'}
  else if (grepl('salvage',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'salvage'}
  else if (grepl('wheel_system',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'wheel_system'}
  else if (grepl('car_origin',bar_label$Coefficient[i])) {
    bar_label$group[i] = 'car_origin'}
}
predictors <- tibble('Predictors' = colnames(m.tr), 'Lasso' = beta.las,
                     'Elastic' = beta.els, 'Ridge' = beta.rid, 'Group' = bar_label$group,
                     'Random Forest' = as.matrix(rf.import[,2]))

ggplot(data = predictors, aes(x = Predictors, y = Lasso, fill = Group)) + 
  geom_bar(stat='identity') + ggtitle('Lasso Coefficients') +
  theme(legend.text = element_text(size = 6)) + 
  scale_fill_manual(values = c('#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A'))
ggplot(data = predictors, aes(x = Predictors, y = Ridge,
                              fill = Group)) + 
  geom_bar(stat='identity')  + ggtitle('Ridge Coefficients') +
  # theme(axis.text.x = element_text(angle=90, size=4, color = 'black'),
  #       legend.text = element_text(size = 6)) + 
  scale_fill_manual(values = c('#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A'))
ggplot(data = predictors, aes(x = Predictors, y = Elastic,
                              fill = Group)) + 
  geom_bar(stat='identity') + ggtitle('Elastic Coefficients') +
  # theme(axis.text.x = element_text(angle=90, size=4, color = 'black'),
  #       legend.text = element_text(size = 6)) + 
  scale_fill_manual(values = c('#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A'))
ggplot(data = predictors, aes(x = Predictors, y = `Random Forest`,
                              fill = Group)) + 
  geom_bar(stat='identity') + ggtitle('Random Forest Importance') +
  # theme(axis.text.x = element_text(angle=90, size=4, color = 'black'),
  #       legend.text = element_text(size = 6)) + 
  scale_fill_manual(values = c('#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A'))
# bar plot for coefficients (ordered)
ggplot(data = predictors, aes(x = reorder(Predictors, -Elastic),
                              y = Lasso, fill = Group)) +
  geom_bar(stat='identity') + ggtitle('Lasso Coefficients') +
  theme(axis.text.x = element_text(angle=90, size=4, color = 'black'),
        legend.text = element_text(size = 6)) +
  scale_fill_manual(values = c('#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A'))
ggplot(data = predictors, aes(x = reorder(Predictors, -Elastic),
                              y = Ridge, fill = Group)) +
  geom_bar(stat='identity')  + ggtitle('Ridge Coefficients') +
  theme(axis.text.x = element_text(angle=90, size=4, color = 'black'),
        legend.text = element_text(size = 6)) +
  scale_fill_manual(values = c('#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A'))
ggplot(data = predictors, aes(x = reorder(Predictors, -Elastic),
                              y = Elastic, fill = Group)) +
  geom_bar(stat='identity') + ggtitle('Elastic Coefficients') +
  theme(axis.text.x = element_text(angle=90, size=4, color = 'black'),
        legend.text = element_text(size = 6)) +
  scale_fill_manual(values = c('#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A'))
ggplot(data = predictors, aes(x = reorder(Predictors, -Elastic),
                              y = `Random Forest`, fill = Group)) +
  geom_bar(stat='identity') + ggtitle('Random Forest Coefficients') +
  theme(axis.text.x = element_text(angle=90, size=4, color = 'black'),
        legend.text = element_text(size = 6)) +
  scale_fill_manual(values = c('#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A',
                               '#808000','#800000','#00FF00','#008000','#FF00FF',
                               '#FF0000','#EAC117','#00FFFF','#808080','#0000FF',
                               '#000000','#ADD8E6','#FFA500','#800080','#A52A2A'))


# box plot for residuals is something like this
# Residual boxplots
{
# I don't know why these are so similar, but I 100% made sure that these are different
# Proof: sum(las.v.resid == rid.v.resid) = 0
# They are very similar box plots, but the individual points ARE different
my.resid <- tibble('Type' = test_or_train,
                   'Lasso' = c(las.v.resid,las.t.resid),
                   'Ridge' = c(rid.v.resid,rid.t.resid),
                   'Elastic' = c(els.v.resid,els.t.resid),
                   'Forest' = c(rf.v.resid,rf.t.resid))
par(mfrow=c(2,2))
boxplot(Lasso~Type, data = my.resid, outline = F, main = 'Lasso Residuals')
boxplot(Ridge~Type, data = my.resid, outline = F, main = 'Ridge Residuals')
boxplot(Elastic~Type, data = my.resid, outline =F, main = 'Elastic Residuals')
boxplot(Forest~Type, data = my.resid, outline = F, main = 'Random Forest Residuals')
par(mfrow=c(1,1))
}


# Trials 2-100
for (i in 2:K) {
  set.seed(i)
  spliter <- sample(1:n, n/5)
  train <- carS[-spliter,]
  test <- carS[spliter,]
  price.tr <- train$price
  price.val <- test$price
  m.tr <- model.matrix(price ~ ., data = train)[,-1]
  m.val <- model.matrix(price ~ ., data = test)[,-1]
  
  # Lasso
  mod.las <- cv.glmnet(m.tr,price.tr, alpha=1) 
  cat(1)
  lambda <- mod.las$lambda.min
  pred.las <- predict(mod.las, s = lambda, newx = m.val)
  tr.las <- predict(mod.las, s = lambda, newx = m.tr)
  v.err[i,1] <- 1 - mean((pred.las - price.val)^2) / mean((mean(pred.las)-price.val)^2)
  t.err[i,1] <- 1 - mean((tr.las - price.tr)^2) / mean((mean(tr.las)-price.tr)^2)
  
  # Ridge
  mod.rid <- cv.glmnet(m.tr,price.tr, alpha=0) 
  lambda <- mod.rid$lambda.min
  pred.rid <- predict(mod.rid, s = lambda, newx = m.val)
  tr.rid <- predict(mod.rid, s = lambda, newx = m.tr)
  v.err[i,2] <- 1- mean((pred.rid- price.val)^2) / mean((mean(pred.rid)-price.val)^2)
  t.err[i,2] <- 1- mean((tr.rid - price.tr)^2) / mean((mean(tr.rid)-price.tr)^2)
  
  # Elastic
  mod.els <- cv.glmnet(m.tr,price.tr, alpha=0.5)
  lambda <- mod.els$lambda.min
  pred.els <- predict(mod.els, s = lambda, newx = m.val)
  tr.els <- predict(mod.els, s = lambda, newx = m.tr)
  v.err[i,3] <- 1 - mean((pred.els- price.val)^2) / mean((mean(pred.els)-price.val)^2)
  t.err[i,3] <- 1 - mean((tr.els - price.tr)^2) / mean((mean(tr.els)-price.tr)^2)
  
  # Random Forest
  rf.car <- randomForest(x=m.tr, y=price.tr,mtry=floor(sqrt(p)), importance=T)
  y.hat.test <- predict(rf.car, newdata = m.val)
  y.hat.train <- predict(rf.car, newdata = m.tr)
  v.err[i,4] <- 1 - mean((y.hat.test-price.val)^2) / mean((mean(y.hat.test)-price.val)^2)
  t.err[i,4] <- 1 - mean((y.hat.train-price.tr)^2) / mean((mean(y.hat.train)-price.tr)^2)
}

# R^2 info
{par(mfrow=c(1,2))
colnames(t.err) <- c('Lasso','Ridge',"Elastic","RF")
colnames(v.err) <- c('Lasso','Ridge',"Elastic","RF")
boxplot(t.err,data = t.err, outline = F, main = 'Training Residuals', ylim = c(.85,.92))
boxplot(v.err,data = v.err, outline = F, main = 'Testing Residuals', ylim = c(.85,.92))
par(mfrow=c(1,1))
}

# 90% CI of R^2
lr2 <- sort(v.err[,1])
c(lr2[6],lr2[94])
rr2 <- sort(v.err[,2])
c(rr2[6],rr2[94])
er2 <- sort(v.err[,3])
c(er2[6],er2[94])
fr2 <- sort(v.err[,4])
c(fr2[6],fr2[94])

