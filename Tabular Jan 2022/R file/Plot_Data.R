# weekdays and weekends
tr <- read.csv('modified_train.csv')
mean(tr[which(tr[,'weekend_1']==1),'num_sold'])
mean(tr[which(tr[,'weekend_1']==0),'num_sold'])

df <- read.csv('train.csv')
############################## each country ####################################
swe <- df[which(df[,2]=='Sweden'),]
plot(1:nrow(swe),swe[,'num_sold'],main = 'sweden',xlab = 'date (from 1/1/2015)',ylab = 'sale')
abline(v=1)
abline(v=365*6)
abline(v=(365+366)*6)
abline(v=(365+366+366)*6)
abline(v=(365+366+366+365)*6)
text(1,1350, as.character(swe[1,'date']), col = "red", adj = c(0, -.1))
text(365*6,1350, as.character(swe[365*6,'date']), col = "red", adj = c(0, -.1))
text((365+366)*6,1350, as.character(swe[(365+366)*6,'date']), col = "red", adj = c(0, -.1))
text((365+366+366)*6,1350, as.character(swe[(365+366+366)*6,'date']), col = "red", adj = c(0, -.1))
text((365+366+366+240)*6,1350, as.character(swe[(365+366+366+364)*6,'date']), col = "red", adj = c(0, -.1))
Fin <- df[which(df[,2]=='Finland'),]
Nor <- df[which(df[,2]=='Norway'),]
plot(1:nrow(Fin),Fin[,'num_sold'],main = 'Finland',xlab = 'date (from 1/1/2015)',ylab = 'sale')
plot(1:nrow(Nor),Nor[,'num_sold'],main = 'Norwar',xlab = 'date (from 1/1/2015)',ylab = 'sale')
################################################################################

########################### Sweden detail ######################################
swe_mart_mug <- swe[which(swe[,'store']=='KaggleMart')[which(swe[,'store']=='KaggleMart')%in%which(swe[,'product']=='Kaggle Mug')],c('date','num_sold')]
swe_rama_mug <- swe[which(swe[,'store']=='KaggleRama')[which(swe[,'store']=='KaggleRama')%in%which(swe[,'product']=='Kaggle Mug')],c('date','num_sold')]
swe_mart_hat <- swe[which(swe[,'store']=='KaggleMart')[which(swe[,'store']=='KaggleMart')%in%which(swe[,'product']=='Kaggle Hat')],c('date','num_sold')]
swe_rama_hat <- swe[which(swe[,'store']=='KaggleRama')[which(swe[,'store']=='KaggleRama')%in%which(swe[,'product']=='Kaggle Hat')],c('date','num_sold')]
swe_mart_stick <- swe[which(swe[,'store']=='KaggleMart')[which(swe[,'store']=='KaggleMart')%in%which(swe[,'product']=='Kaggle Sticker')],c('date','num_sold')]
swe_rama_stick <- swe[which(swe[,'store']=='KaggleRama')[which(swe[,'store']=='KaggleRama')%in%which(swe[,'product']=='Kaggle Sticker')],c('date','num_sold')]
########################### Finland detail ######################################
fin_mart_mug <- Fin[which(Fin[,'store']=='KaggleMart')[which(Fin[,'store']=='KaggleMart')%in%which(Fin[,'product']=='Kaggle Mug')],c('date','num_sold')]
fin_rama_mug <- Fin[which(Fin[,'store']=='KaggleRama')[which(Fin[,'store']=='KaggleRama')%in%which(Fin[,'product']=='Kaggle Mug')],c('date','num_sold')]
fin_mart_hat <- Fin[which(Fin[,'store']=='KaggleMart')[which(Fin[,'store']=='KaggleMart')%in%which(Fin[,'product']=='Kaggle Hat')],c('date','num_sold')]
fin_rama_hat <- Fin[which(Fin[,'store']=='KaggleRama')[which(Fin[,'store']=='KaggleRama')%in%which(Fin[,'product']=='Kaggle Hat')],c('date','num_sold')]
fin_mart_stick <- Fin[which(Fin[,'store']=='KaggleMart')[which(Fin[,'store']=='KaggleMart')%in%which(Fin[,'product']=='Kaggle Sticker')],c('date','num_sold')]
fin_rama_stick <- Fin[which(Fin[,'store']=='KaggleRama')[which(Fin[,'store']=='KaggleRama')%in%which(Fin[,'product']=='Kaggle Sticker')],c('date','num_sold')]
########################### Norway detail ######################################
nor_mart_mug <- Nor[which(Nor[,'store']=='KaggleMart')[which(Nor[,'store']=='KaggleMart')%in%which(Nor[,'product']=='Kaggle Mug')],c('date','num_sold')]
nor_rama_mug <- Nor[which(Nor[,'store']=='KaggleRama')[which(Nor[,'store']=='KaggleRama')%in%which(Nor[,'product']=='Kaggle Mug')],c('date','num_sold')]
nor_mart_hat <- Nor[which(Nor[,'store']=='KaggleMart')[which(Nor[,'store']=='KaggleMart')%in%which(Nor[,'product']=='Kaggle Hat')],c('date','num_sold')]
nor_rama_hat <- Nor[which(Nor[,'store']=='KaggleRama')[which(Nor[,'store']=='KaggleRama')%in%which(Nor[,'product']=='Kaggle Hat')],c('date','num_sold')]
nor_mart_stick <- Nor[which(Nor[,'store']=='KaggleMart')[which(Nor[,'store']=='KaggleMart')%in%which(Nor[,'product']=='Kaggle Sticker')],c('date','num_sold')]
nor_rama_stick <- Nor[which(Nor[,'store']=='KaggleRama')[which(Nor[,'store']=='KaggleRama')%in%which(Nor[,'product']=='Kaggle Sticker')],c('date','num_sold')]
################################################################################

############################### Kaggle Mug #####################################
#### sweden mart mug####
plot(1:nrow(swe_mart_mug),swe_mart_mug[,'num_sold'],main = 'swe_mart_mug',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(swe_mart_mug[1:365,'num_sold'])
d2 <- 365+which.min(swe_mart_mug[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(swe_mart_mug[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(swe_mart_mug[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,350, as.character(swe_mart_mug[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,350, as.character(swe_mart_mug[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,350, as.character(swe_mart_mug[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,350, as.character(swe_mart_mug[d4,'date']), col = "red", adj = c(0, -.1))
####### sweden rama mug ####
plot(1:nrow(swe_rama_mug),swe_rama_mug[,'num_sold'],main = 'swe_rama_mug',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(swe_rama_mug[1:365,'num_sold'])
d2 <- 365+which.min(swe_rama_mug[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(swe_rama_mug[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(swe_rama_mug[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,700, as.character(swe_rama_mug[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,700, as.character(swe_rama_mug[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,700, as.character(swe_rama_mug[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,700, as.character(swe_rama_mug[d4,'date']), col = "red", adj = c(0, -.1))
####### finland rama mug ####
plot(1:nrow(fin_rama_mug),fin_rama_mug[,'num_sold'],main = 'fin_rama_mug',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(fin_rama_mug[1:365,'num_sold'])
d2 <- 365+which.min(fin_rama_mug[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(fin_rama_mug[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(fin_rama_mug[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,700, as.character(fin_rama_mug[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,700, as.character(fin_rama_mug[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,700, as.character(fin_rama_mug[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,700, as.character(fin_rama_mug[d4,'date']), col = "red", adj = c(0, -.1))
####### finland mart mug ####
plot(1:nrow(fin_mart_mug),fin_mart_mug[,'num_sold'],main = 'fin_mart_mug',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(fin_mart_mug[1:365,'num_sold'])
d2 <- 365+which.min(fin_mart_mug[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(fin_mart_mug[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(fin_mart_mug[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,700, as.character(fin_mart_mug[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,700, as.character(fin_mart_mug[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,700, as.character(fin_mart_mug[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,700, as.character(fin_mart_mug[d4,'date']), col = "red", adj = c(0, -.1))
####### nowway mart mug ####
plot(1:nrow(nor_mart_mug),nor_mart_mug[,'num_sold'],main = 'nor_mart_mug',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(nor_mart_mug[1:365,'num_sold'])
d2 <- 365+which.min(nor_mart_mug[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(nor_mart_mug[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(nor_mart_mug[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,700, as.character(nor_mart_mug[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,700, as.character(nor_mart_mug[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,700, as.character(nor_mart_mug[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,700, as.character(nor_mart_mug[d4,'date']), col = "red", adj = c(0, -.1))
####### nowway rama mug ####
plot(1:nrow(nor_rama_mug),nor_rama_mug[,'num_sold'],main = 'nor_rama_mug',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(nor_rama_mug[1:365,'num_sold'])
d2 <- 365+which.min(nor_rama_mug[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(nor_rama_mug[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(nor_rama_mug[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,700, as.character(nor_rama_mug[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,700, as.character(nor_rama_mug[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,700, as.character(nor_rama_mug[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,700, as.character(nor_rama_mug[d4,'date']), col = "red", adj = c(0, -.1))
#### comparasion ####
plot(1:1461,swe_mart_mug[,2]/swe_rama_mug[,2],xlab = 'date',ylab = 'value',main = 'swe_mat_mug/swe_rama_mug')
plot(1:1461,fin_mart_mug[,2]/fin_rama_mug[,2],xlab = 'date',ylab = 'value',main = 'fin_mat_mug/fin_rama_mug')
plot(1:1461,nor_mart_mug[,2]/nor_rama_mug[,2],xlab = 'date',ylab = 'value',main = 'nor_mat_mug/nor_rama_mug')
################################################################################

############################### Kaggle Hat #####################################
#### sweden mart hat####
plot(1:nrow(swe_mart_hat),swe_mart_hat[,'num_sold'],main = 'swe_mart_hat',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(swe_mart_hat[1:365,'num_sold'])
d2 <- 365+which.min(swe_mart_hat[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(swe_mart_hat[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(swe_mart_hat[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,750, as.character(swe_mart_hat[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,750, as.character(swe_mart_hat[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,750, as.character(swe_mart_hat[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,750, as.character(swe_mart_hat[d4,'date']), col = "red", adj = c(0, -.1))
####### sweden rama hat####
plot(1:nrow(swe_rama_hat),swe_rama_hat[,'num_sold'],main = 'swe_rama_hat',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(swe_rama_hat[1:365,'num_sold'])
d2 <- 365+which.min(swe_rama_hat[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(swe_rama_hat[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(swe_rama_hat[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,1700, as.character(swe_rama_hat[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,1700, as.character(swe_rama_hat[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,1700, as.character(swe_rama_hat[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,1700, as.character(swe_rama_hat[d4,'date']), col = "red", adj = c(0, -.1))
#### norway mart hat####
plot(1:nrow(nor_mart_hat),nor_mart_hat[,'num_sold'],main = 'nor_mart_hat',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(nor_mart_hat[1:365,'num_sold'])
d2 <- 365+which.min(nor_mart_hat[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(nor_mart_hat[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(nor_mart_hat[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,750, as.character(nor_mart_hat[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,750, as.character(nor_mart_hat[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,750, as.character(nor_mart_hat[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,750, as.character(nor_mart_hat[d4,'date']), col = "red", adj = c(0, -.1))
####### norway rama hat ####
plot(1:nrow(nor_rama_hat),nor_rama_hat[,'num_sold'],main = 'nor_rama_hat',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(nor_rama_hat[1:365,'num_sold'])
d2 <- 365+which.min(nor_rama_hat[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(nor_rama_hat[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(nor_rama_hat[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,1700, as.character(nor_rama_hat[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,1700, as.character(nor_rama_hat[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,1700, as.character(nor_rama_hat[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,1700, as.character(nor_rama_hat[d4,'date']), col = "red", adj = c(0, -.1))
#### Finland mart hat####
plot(1:nrow(fin_mart_hat),fin_mart_hat[,'num_sold'],main = 'fin_mart_hat',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(fin_mart_hat[1:365,'num_sold'])
d2 <- 365+which.min(fin_mart_hat[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(fin_mart_hat[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(fin_mart_hat[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,750, as.character(fin_mart_hat[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,750, as.character(fin_mart_hat[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,750, as.character(fin_mart_hat[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,750, as.character(fin_mart_hat[d4,'date']), col = "red", adj = c(0, -.1))
####### Finland rama hat ####
plot(1:nrow(fin_rama_hat),fin_rama_hat[,'num_sold'],main = 'fin_rama_hat',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(fin_rama_hat[1:365,'num_sold'])
d2 <- 365+which.min(fin_rama_hat[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(fin_rama_hat[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(fin_rama_hat[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,1700, as.character(fin_rama_hat[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,1700, as.character(fin_rama_hat[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,1700, as.character(fin_rama_hat[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,1700, as.character(fin_rama_hat[d4,'date']), col = "red", adj = c(0, -.1))
#### comparasion ####
plot(1:1461,swe_mart_hat[,2]/swe_rama_hat[,2],xlab = 'date',ylab = 'value',main = 'swe_mat_hat/swe_rama_hat')
plot(1:1461,fin_mart_hat[,2]/fin_rama_hat[,2],xlab = 'date',ylab = 'value',main = 'fin_mat_hat/fin_rama_hat')
plot(1:1461,nor_mart_hat[,2]/nor_rama_hat[,2],xlab = 'date',ylab = 'value',main = 'nor_mat_hat/nor_rama_hat')
################################################################################

############################### Kaggle Sticker #####################################
#### sweden mart stick####
plot(1:nrow(swe_mart_stick),swe_mart_stick[,'num_sold'],main = 'swe_mart_stick',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(swe_mart_stick[1:365,'num_sold'])
d2 <- 365+which.min(swe_mart_stick[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(swe_mart_stick[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(swe_mart_stick[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,222, as.character(swe_mart_stick[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,222, as.character(swe_mart_stick[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,222, as.character(swe_mart_stick[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,222, as.character(swe_mart_stick[d4,'date']), col = "red", adj = c(0, -.1))
####### sweden rama stick####
plot(1:nrow(swe_rama_stick),swe_rama_stick[,'num_sold'],main = 'swe_rama_stick',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(swe_rama_stick[1:365,'num_sold'])
d2 <- 365+which.min(swe_rama_stick[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(swe_rama_stick[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(swe_rama_stick[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,500, as.character(swe_rama_stick[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,500, as.character(swe_rama_stick[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,500, as.character(swe_rama_stick[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,500, as.character(swe_rama_stick[d4,'date']), col = "red", adj = c(0, -.1))
#### norway mart stick####
plot(1:nrow(nor_mart_stick),nor_mart_stick[,'num_sold'],main = 'nor_mart_stick',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(nor_mart_stick[1:365,'num_sold'])
d2 <- 365+which.min(nor_mart_stick[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(nor_mart_stick[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(nor_mart_stick[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,350, as.character(nor_mart_stick[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,350, as.character(nor_mart_stick[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,350, as.character(nor_mart_stick[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,350, as.character(nor_mart_stick[d4,'date']), col = "red", adj = c(0, -.1))
####### norway rama stick ####
plot(1:nrow(nor_rama_stick),nor_rama_stick[,'num_sold'],main = 'nor_rama_hat',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(nor_rama_stick[1:365,'num_sold'])
d2 <- 365+which.min(nor_rama_stick[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(nor_rama_stick[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(nor_rama_stick[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,700, as.character(nor_rama_stick[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,700, as.character(nor_rama_stick[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,700, as.character(nor_rama_stick[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,700, as.character(nor_rama_stick[d4,'date']), col = "red", adj = c(0, -.1))
#### Finland mart stick####
plot(1:nrow(fin_mart_stick),fin_mart_stick[,'num_sold'],main = 'fin_mart_stick',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(fin_mart_stick[1:365,'num_sold'])
d2 <- 365+which.min(fin_mart_stick[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(fin_mart_stick[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(fin_mart_stick[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,250, as.character(fin_mart_stick[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,250, as.character(fin_mart_stick[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,250, as.character(fin_mart_stick[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,250, as.character(fin_mart_stick[d4,'date']), col = "red", adj = c(0, -.1))
####### Finland rama stick ####
plot(1:nrow(fin_rama_stick),fin_rama_stick[,'num_sold'],main = 'fin_rama_stick',xlab = 'date(start from 1/1/2015)',ylab = 'sale')
d1 <- which.min(fin_rama_stick[1:365,'num_sold'])
d2 <- 365+which.min(fin_rama_stick[366:(366+366),'num_sold'])
d3 <- 365+366+which.min(fin_rama_stick[(366+366):(366+366+365),'num_sold'])
d4 <- 365+366+365+which.min(fin_rama_stick[(366+366+365):(366+366+365+365),'num_sold'])
abline(v=d1)
abline(v=d2)
abline(v=d3)
abline(v=d4)
text(d1,500, as.character(fin_rama_stick[d1,'date']), col = "red", adj = c(0, -.1))
text(d2,500, as.character(fin_rama_stick[d2,'date']), col = "red", adj = c(0, -.1))
text(d3,500, as.character(fin_rama_stick[d3,'date']), col = "red", adj = c(0, -.1))
text(d4,500, as.character(fin_rama_stick[d4,'date']), col = "red", adj = c(0, -.1))
#### comparasion ####
plot(1:1461,swe_mart_stick[,2]/swe_rama_stick[,2],xlab = 'date',ylab = 'value',main = 'swe_mat_stick/swe_rama_stick')
plot(1:1461,fin_mart_stick[,2]/fin_rama_stick[,2],xlab = 'date',ylab = 'value',main = 'fin_mat_stick/fin_rama_stick')
plot(1:1461,nor_mart_stick[,2]/nor_rama_stick[,2],xlab = 'date',ylab = 'value',main = 'nor_mat_stick/nor_rama_stick')
























