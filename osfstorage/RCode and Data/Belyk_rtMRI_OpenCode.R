library(zoo)
library(fda)
library(lattice)
library(cluster)
library(tidyverse)
library(ggplot2)
library(ggridges)
library(abind)
library(lme4)
library(lmerTest)
library(RColorBrewer)
library(Rtsne)
library(arm)

setwd("/yourwd") #put your working directory here

#save.image("Belyk_rtMRI_OpenCode.Rdata")
load("Belyk_rtMRI_OpenCode.Rdata")

###############
###functions###
###############
substrRight <- function(x, n){ #get last n characters
  substr(x, nchar(x)-n+1, nchar(x))
}

substrEnd<-function(x,n){ #get characters from n to end
  substr(x,n, length(x))
}

ranger<-function(dat){max(dat)-min(dat)}

###custom ggplot theme
theme_mb<-function(){
  theme_light(base_size=16) %+replace% 
    theme(strip.background = element_rect(fill="grey95"),
          strip.text.x = element_text(colour="black"),
          strip.text.y = element_text(colour="black", angle=270))
}

############################
###CUSTOM FUNCTION: fPCA ###
############################
#doesn't partition Y-Z, I prefer them combined
fpca_MB<-function(fd_in, no_components=NA){ #returns a list with 1) an fd object with components, 2) pca scores for each obs
  if(is.na(no_components)){no_components=dim(fd_in$coefs)[1]*2}
  
  #reshape coefs for pcs inputs
  ycoefs<-t(fd_in$coefs[,,1])
  zcoefs<-t(fd_in$coefs[,,2])
  coefs<-cbind(ycoefs,zcoefs)
  fPCA<-prcomp(coefs, center=T, scale=T, rank=no_components)
  
  #Proportion of variance & scree plot
  print(paste("Propotion of Variance explained by components 1 to",no_components))
  vars<-(fPCA$sdev^2)[1:no_components]
  scree_values <- vars/sum(vars)
  plot(scree_values, #plot Scree
       type="b", main="Scree Plot", xlab = "Component", ylab="Proportion of Variation")
  print(scree_values)
  
  fPCA_scores<-fPCA$x #get scores
  
  ###build a new fd objects
  pca_fd<-fd_in #copy over fd object and all of it's properties
  no_coefs<-dim(pca_fd$coefs)[1]
  pca_fd$coefs<-array(dim=c(no_coefs,no_components,2))
  
  y_loadings <- fPCA$rotation[1:no_coefs,]
  z_loadings <- fPCA$rotation[(no_coefs+1):(2*no_coefs),]
  all_loadings <- abind(y_loadings,z_loadings, along=3)
  
  pca_fd$coefs<-all_loadings #put Y loadings in place
  pca_fd$fdnames$reps<-paste("Component",1:no_components, sep="_")     
  
  return(list(pca_fd,fPCA_scores))
}

##################################################################
###CUSTOM FUCTION: PRETTY FIGURE FOR PLOTTING VOCAL TRACT SPACE###
##################################################################
plot_fpc<-function(ymean,zmean, #objects for drawing the mean VT shape
                   referenceY=ymean[1],referenceZ=zmean[1], #where should the plot origin be. By default whereever the trace data starts
                   ypc, zpc, #the principal component data
                   scores=c(1,2,3,4), #plot contours at these PC scores
                   invertY=F,invertZ=F, #does either axis need inverting
                   ylim= c(-25,30), xlim = c(0,55), #axis limits
                   col_palette="RdBu", #select diverging pallet from display.brewer.all(type="div")
                   xlab="Y",ylab="Z", main="Principal component of morphological variation",
                   ...){ #plot title
  
  #invert Y axis if requested, not well tested
  flipY<-ifelse(invertY, -1, 1) 
  referenceY=referenceY*flipY
  ymean=ymean*flipY
  ypc=ypc*flipY
  
  #invert Z axis if requested, not well tested
  flipZ<-ifelse(invertZ, -1, 1) 
  referenceZ=referenceZ*flipZ
  zmean=zmean*flipZ
  zpc=zpc*flipZ
  
  plot(abs(ymean-referenceY),zmean-referenceZ #plot mean curves, centered at first value which is most reliable point for anchor
       ,ylim=  ylim
       ,xlim = xlim
       ,type='l'
       ,pch=16
       ,xlab = xlab
       ,ylab = ylab
       ,main = main
       ,cex.lab = 2
       ,cex.axis = 2)
  
  brewpal<-brewer.pal(n=length(scores)*2, name=col_palette) #this sets us at max 10 colours
  reds=brewpal[length(scores):1]
  blues=brewpal[(length(scores)+1):(length(scores)*2)]
  
  #layers of polygon components, lowest to highest layer, origin on tongue root, showing reconstructions of fPCs, -() to reorient axis
  iter=1
  for(score in rev(scores)){ #better to count down
    polygon((ymean-ypc*score - (referenceY-ypc*score)[1]),zmean-zpc*score - (referenceZ-zpc*score)[1], lty=0, col=blues[iter]) 
    polygon((ymean+ypc*score - (referenceY+ypc*score)[1]),zmean+zpc*score - (referenceZ+zpc*score)[1], lty=0, col=reds[iter])
    iter=iter+1 #this is needed in case the requested scores are not just a sequence up from 1
  }
  polygon((ymean-referenceY),zmean-referenceZ, col="black") #top layer with mean
  
  #outlines for continuations
  iter=1
  for(score in rev(scores)){
    lines((ymean-ypc*score - (referenceY-ypc*score)[1]),zmean-zpc*score - (referenceZ-zpc*score)[1], lty=2, col=blues[iter]) 
    lines((ymean+ypc*score - (referenceY+ypc*score)[1]),zmean+zpc*score - (referenceZ+zpc*score)[1], lty=2, col=reds[iter])
    iter=iter+1
  }
  
  legend("topright", legend=paste(scores,"X fPC"), fill=rev(reds), cex=1.1)
  legend("topleft", legend=paste(-scores,"X fPC"), fill=rev(blues), cex=1.1)
  legend("top", legend=c("Mean Shape"), fill=c("black"), cex=0.8)
}

# the following is my data input, wrangling and fda modelling protocol
# input files are not provided as they could not be readily anonimised
# the .Rdata file provided includes wrangled data to support later code chunks

# #################
# ###DATA IMPORT### makes strong assumptions about matcghing filenames that hold for morpho_masking.m pipeline
# #################
# #where data are
# data_dir = "data_vt/"
# logs_dir = "logs_vt/"
# input_filenames_X <-list.files(path=data_dir, pattern="*_X.csv")
# input_filenames_Y <-list.files(path=data_dir, pattern="*_Y.csv")
# log_filenames <-list.files(path=logs_dir, pattern="*_log.csv")
# demographics<- read_csv("demographics/demographics.csv")
# #in Matlab I've used X-Y to indicte the rows and columns of amatrix
# #here I will convert ot Y-Z to indicate orientation within the scanner
# 
# #make placeholders
# listy_list_Y<-list()
# listy_list_Z<-list()
# listy_list_T <- list()
# 
# n_frames_list<-list()
# 
# listy_list_center_Y<-list()
# listy_list_center_Z<-list()
# 
# list_logs<-list()
# contour_size<-list()
# ###iterate through frames in each file
# ###
# 
# no_runs<-length(input_filenames_X)
# for(run in 1:no_runs){ #loop through files
#   print(paste("Finished run", as.character(run), "of", as.character(no_runs))) #note on progress
#   
#   n_frames_list[[run]] <- length(readLines(paste(data_dir,input_filenames_X[run], sep="")))
#   
#   #placeholder within lists
#   listy_list_Y[[run]] <- list()
#   listy_list_Z[[run]] <- list()
#   
#   listy_list_T[[run]] <- list()
#   
#   listy_list_center_Y[[run]] <- list()
#   listy_list_center_Z[[run]] <- list()
#   
#   contour_size[[run]]<-list()
#   
#   list_logs[[run]]<-read_csv(paste(logs_dir,log_filenames[run], sep="")) #####
#   Run_name<-str_remove(log_filenames[run], pattern="_log.csv")
#   list_logs[[run]]$"Run"<-Run_name
#   list_logs[[run]]$"Participant"<- substr(Run_name,1, nchar(Run_name)-7) #remove block0
#   list_logs[[run]]$"Block"      <- substrRight(Run_name,6) #keep only block0
#   
#   for(frame in 1:n_frames_list[[run]]){ #loop through frames in files
#     frames_to_skip=frame-1
#     
#     listy_list_Y[[run]][[frame]] <-  scan(paste(data_dir,input_filenames_X[run], sep=""), skip=frames_to_skip, nlines=1, sep=",")
#     listy_list_Z[[run]][[frame]] <-  scan(paste(data_dir,input_filenames_Y[run], sep=""), skip=frames_to_skip, nlines=1, sep=",")
#     
#     #plot(listy_list_Y[[run]][[frame]],listy_list_Z[[run]][[frame]], type="l") #have a look. it's upside because matlab
#     contour_size[[run]][[frame]] <- length(listy_list_Y[[run]][[frame]]) #should be same as if taken from Z
#     listy_list_T[[run]][[frame]] <- 1:contour_size[[run]][[frame]] #innumerate points, useful later
#     
#     #center, fda assumes contours are centered around 0
#     listy_list_center_Y[[run]][[frame]] <- listy_list_Y[[run]][[frame]]-mean(listy_list_Y[[run]][[frame]])
#     listy_list_center_Z[[run]][[frame]] <- listy_list_Z[[run]][[frame]]-mean(listy_list_Z[[run]][[frame]])
#     
#     listy_list_center_Z[[run]][[frame]] <- listy_list_center_Z[[run]][[frame]]*(-1) #reflect about Z axis
#     #plot(listy_list_center_Y[[run]][[frame]],listy_list_center_Z[[run]][[frame]], type="l")
#     
#   }
# }
# 
# ##################
# ###DATA WRANGLE###
# ##################
# 
# ###flatten listy_lists to single tiered list here
# list_Y<-flatten(listy_list_Y)
# list_Z<-flatten(listy_list_Z)
# list_T<-flatten(listy_list_T)
# 
# list_center_Y<-flatten(listy_list_center_Y)
# list_center_Z<-flatten(listy_list_center_Z)
# 
# #flaten list_logs to tibble
# logs<-list_logs[[1]] #initialise. put first log in
# for(l in 2:length(list_logs)){ #add more starting with the second
#   logs<-rbind(logs, list_logs[[l]])
# }  
# 
# ###Make column for words 
# logs$Word<-NA
# logs$Word<-substr(logs$Sound_File,2,5)
# unique(logs$Word) #some ballsed up ones need a fix
# 
# #fix bad encoding
# logs<-logs %>%
#   mutate(Word=str_replace(Word,'ARD_','BARD')) %>%
#   mutate(Word=str_replace(Word,'EAD_','BEAD'))
# 
# ###Make columns for vocal tract length and f0
# #fix bad encoding
# unique(logs$Voice_Type)
# logs[logs$Voice_Type=="?",] #four instances, two Trial_type 5, two Trials type 9 all same participant
# logs[logs$Trial_Type==5,] #CHILD 
# logs[logs$Trial_Type==9,] #MALE
# 
# logs[logs$Voice_Type=="?" & logs$Trial_Type==5,"Voice_Type"] <- "CHILD"
# logs[logs$Voice_Type=="?" & logs$Trial_Type==9,"Voice_Type"] <- "MALE"
# unique(logs$Voice_Type)
# 
# #Code useful tidy colums for pitch and vocal tract length
# logs<-logs %>%
#   mutate(Pitch = recode(Voice_Type, 
#                         "CHILD"="High", 
#                         "NORMAL"="Normal",
#                         "SMALL LOW"="Low",
#                         "LARGE HIGH" = "High",
#                         "MALE" = "Low")) %>%
#   mutate(VTL = recode(Voice_Type, 
#                       "CHILD"="Short", 
#                       "NORMAL"="Normal",
#                       "SMALL LOW"="Short",
#                       "LARGE HIGH" = "Long",
#                       "MALE" = "Long"))
# 
# #I had to do some checking on the demographics file and fix some Participant code typos
# tmp<-left_join(logs, demographics, by="Participant") #check that I've got this function right
# sum(is.na(tmp))
# unique(tmp[rowSums(is.na(tmp)) > 0,]$"Participant")
# rm("tmp")
# logs<-left_join(logs, demographics, by="Participant") #add dempographic info to logs
# 
# #add anonymised ID label
# logs$ID<-factor(logs$Participant, labels=paste("P",sample(1:length(unique(logs$Participant))),sep=""))
# 
# 
# total_frames <- dim(logs)[1] 
# 
# #########
# ###GCV### & initial un-time-normalized fd model
# #########
# #begin smoothing iterations
# y_list_fd  = list()
# z_list_fd  = list()
# y_lambda_list = list()
# z_lambda_list = list()
# 
# for (i in 1:total_frames) { #loop through all frames, subjects is a vector that marks which subject contributed each frame
#   t = list_T[[i]]
#   y = list_center_Y[[i]]
#   z = list_center_Z[[i]]
#   rng <- range(t)
#   knots <- t
#   norder <- 4
#   nbasis <- length(knots) + norder - 2
#   basis_i <- create.bspline.basis(rng, nbasis, norder, knots) # note that each basis is different
#   Lfdobj <- 2
#   
#   #GCV chooses lambda (roughness penalty) for curve i
#   rng = range(t)
#   knots = t
#   norder = 4
#   nbasis = length(knots) + norder - 2
#   basis_i = create.bspline.basis(rng, nbasis, norder, knots)
#   Lfdobj = 2
#   loglam = seq(-10,0,1) # the range of roughness penalties to test
#   nlam = length(loglam)
#   y_dfsave = rep(NA,nlam)
#   z_dfsave = rep(NA,nlam)
#   y_gcvsave = rep(NA,nlam)
#   z_gcvsave = rep(NA,nlam)
#   
#   for (ilam in 1:nlam) {
#     cat(paste('log10 lambda =',loglam[ilam],'\n'))
#     lambda1 = 10^loglam[ilam]
#     fdPar_i = fdPar(basis_i, Lfdobj, lambda1)
#     y_smoothlist = smooth.basis(t,y,fdPar_i)
#     z_smoothlist = smooth.basis(t,z,fdPar_i)
#     y_dfsave[ilam] = y_smoothlist$df
#     z_dfsave[ilam] = z_smoothlist$df
#     y_gcvsave[ilam] = sum(y_smoothlist$gcv)
#     z_gcvsave[ilam] = sum(z_smoothlist$gcv)
#     
#   }
#   y_lambda = loglam[which.min(y_gcvsave)]
#   z_lambda = loglam[which.min(z_gcvsave)]
#   y_lambda_list[[i]] = 10^y_lambda
#   z_lambda_list[[i]] = 10^z_lambda
#   
#   fdPar_yi <- fdPar(basis_i, Lfdobj, y_lambda_list[[i]]) #choose some stock settings to apply across.
#   fdPar_zi <- fdPar(basis_i, Lfdobj, z_lambda_list[[i]]) #choose some stock settings to apply across.
#   
#   y_list_fd[[i]] <- smooth.basis(t,y,fdPar_yi)$fd #y needs to
#   z_list_fd[[i]] <- smooth.basis(t,z,fdPar_zi)$fd #y needs to
# }
# 
# ####################
# ###Time normalize###
# ####################
# len<-unlist(unlist(contour_size))
# n_points = median(len) #number of points for normalized time z#consider median?
# n_sample = length(y_list_fd)
# # n_points must be an odd number
# a = round(n_points/2) #rounded
# b = n_points/2        #unrounded
# if (a == b) n_points = n_points +1 #if unrounded and rounded are same, is even 
# 
# t_norm <- (0:(n_points-1))/(n_points-1) # normalized time points
# y  = matrix(nrow = n_points, ncol = n_sample)
# z  = matrix(nrow = n_points, ncol = n_sample)
# 
# for (i in 1:n_sample) {
#   y[,i]  <-  eval.fd(seq(1, len[i], length.out=n_points), y_list_fd[[i]]) 
#   z[,i]  <-  eval.fd(seq(1, len[i], length.out=n_points), z_list_fd[[i]]) 
#   # may need to multiply evals by 0.999 to correct rounding errors in the last sample
# }
# 
# #combine y and z coords
# yz <- abind(y,z,along=3) 
# dim(yz) #trace_angle, frame, pixel 
# 
# 
# #create fd, different from above
# norm_range = c(0,1)
# knots <- t_norm[seq(1,n_points,2)]
# norder <- 4
# nbasis <- length(knots) + norder - 2
# norm_basis <- create.bspline.basis(norm_range, nbasis, norder, knots)
# Lfdobj <- 2
# lambda_weak <- 1e-16  # deliberately weak smoothing since data have already been smoothed
# norm_Par <- fdPar(norm_basis, Lfdobj, lambda_weak)
# yz_fd  <- smooth.basis(t_norm,yz,norm_Par)$fd
# yz_fd$fdnames[[3]] <-list("Y", "Z") #more useful names
# plot(yz_fd)

# Replacing Participant (which contains identifying info) with anonymized labels
#PartipID<-logs[match(unique(logs$Participant),logs$Participant), c("Participant", "ID")]
#demographics<-left_join(demographics, PartipID, by="Participant")
#demographics<-demographics %>% dplyr::select(-Participant)
#logs<-logs %>% dplyr::select(-Participant)
#rm(PartipID)

##########
###fPCA###
##########
pca_fd<-fpca_MB(yz_fd, 15) #chose 5 from the scree plots
pca_fd[[1]] #fd object with loadings
pca_fd[[2]]  #scores

evals<-seq(0, 0.99, 0.01)
ymean<-rowMeans(eval.fd(evals, yz_fd)[,,1]) #mean curve
zmean<-rowMeans(eval.fd(evals, yz_fd)[,,2]) #mean curve

###use custom plot function to see how well that worked
#first fPC
ypc1<-eval.fd(evals, pca_fd[[1]][1,1])
zpc1<-eval.fd(evals, pca_fd[[1]][1,2])

#second fPC
ypc2<-eval.fd(evals, pca_fd[[1]][2,1])
zpc2<-eval.fd(evals, pca_fd[[1]][2,2])

#third fPC
ypc3<-eval.fd(evals, pca_fd[[1]][3,1])
zpc3<-eval.fd(evals, pca_fd[[1]][3,2])

#fourth fPC
ypc4<-eval.fd(evals, pca_fd[[1]][4,1])
zpc4<-eval.fd(evals, pca_fd[[1]][4,2])

#fifth fPC
ypc5<-eval.fd(evals, pca_fd[[1]][5,1])
zpc5<-eval.fd(evals, pca_fd[[1]][5,2])

pdf("plots/VT_Shape_fPC1.pdf") #flat/shallow vs deep/bendy vocal tracts? maybe something about ID here?
plot_fpc(ymean,zmean,ypc=ypc1,zpc=zpc1, scores=c(5,10,15,20),ylim= c(-25,20), xlim = c(0,45),
         main="functional Principal Component 1", col_palette = "PuOr",
         xlab="Y (pixels from lip aperture)", ylab= "Z (pixels from lip aperture)")
dev.off()

###THIS ONE
pdf("plots/VT_Shape_fPC2.pdf") #flat/shallow vs deep/bendy vocal tracts? maybe something about ID here?
plot_fpc(ymean,zmean,ypc=ypc2,zpc=zpc2, scores=c(5,10,15,20),ylim= c(-25,20), xlim = c(0,45),
         main="functional Principal Component 2", col_palette = "PuOr",
         xlab="Y (pixels from lip aperture)", ylab= "Z (pixels from lip aperture)")
dev.off()

pdf("plots/VT_Shape_fPC3.pdf") #flat/shallow vs deep/bendy vocal tracts? maybe something about ID here?
plot_fpc(ymean,zmean,ypc=ypc3,zpc=zpc3, scores=c(5,10,15,20), 
           main="VT Shape fPC3 (9.7% of variance)", col_palette = "PuOr",
         xlab="Y (pixels from lip aperture)", ylab= "Z (pixels from lip aperture)")
dev.off()

pdf("plots/VT_Shape_fPC4.pdf") #flat/shallow vs deep/bendy vocal tracts? maybe something about ID here?
plot_fpc(ymean,zmean,ypc=ypc4,zpc=zpc4, scores=c(5,10,15,20),
         main="VT Shape fPC4 (8.6% of variance)", col_palette = "PuOr",
         xlab="Y (pixels from lip aperture)", ylab= "Z (pixels from lip aperture)")
dev.off()

pdf("plots/VT_Shape_fPC5.pdf") #flat/shallow vs deep/bendy vocal tracts? maybe something about ID here?
plot_fpc(ymean,zmean,ypc=ypc5,zpc=zpc5, scores=c(5,10,15,20),
         main="VT Shape fPC5 (3.5% of variance)", col_palette = "PuOr",
         xlab="Y (pixels from lip aperture)", ylab= "Z (pixels from lip aperture)")
dev.off()

pca_dat<-as_tibble(cbind(pca_fd[[2]],logs))
ggplot(data=pca_dat, aes(PC1, PC2, colour=VTL))+
  geom_point()

ggplot(data=pca_dat, aes(PC1, PC2, colour=Word))+
  geom_point()

ggplot(data=pca_dat, aes(VTL, PC2, colour=VTL, fill=VTL))+
  geom_violin()

ggplot(data=pca_dat, aes(Word, PC1))+
  geom_boxplot()

###Identify good and poor performers###
#make summary for median and CI for each participant condition
pca_grp<-pca_dat %>%
  mutate(ID=as.factor(ID)) %>%
  group_by(ID, Sex, VTL) %>% #order participants by Sev
  summarise(PC1_Median=median(PC1), PC1_CI_lower=quantile(PC1,probs=0.025),PC1_CI_upper=quantile(PC1,probs=0.975),
            PC2_Median=median(PC2), PC2_CI_lower=quantile(PC2,probs=0.025),PC2_CI_upper=quantile(PC2,probs=0.975),
            PC3_Median=median(PC3), PC3_CI_lower=quantile(PC3,probs=0.025),PC3_CI_upper=quantile(PC3,probs=0.975),
            PC4_Median=median(PC4), PC4_CI_lower=quantile(PC4,probs=0.025),PC4_CI_upper=quantile(PC4,probs=0.975),
            PC5_Median=median(PC5), PC5_CI_lower=quantile(PC5,probs=0.025),PC5_CI_upper=quantile(PC5,probs=0.975))

#measure range for each participant. This seems like a good metric of performance quality to me
pca_rngs<-pca_dat %>%
  mutate(ID=as.factor(ID)) %>%
  group_by(ID,Sex) %>% #order participants by Sev
  summarise(PC1_range=ranger(PC1),
            PC2_range=ranger(PC2))

pca_ploting<-left_join(pca_grp,pca_rngs,by=c("ID", "Sex")) #combine

#order factors for prettier plotting
pca_ploting<- ungroup(pca_ploting) %>% #ungroup so I can fiddle factors
  arrange(Sex, PC2_range) 
pca_ploting$ID<- fct_inorder(pca_ploting$ID) #reorder factors by first appearance

pca_ploting %>% #filter(VTL != "Normal") %>%
  ggplot(aes(x=PC2_Median, y=ID, color=Sex))+
  geom_point(aes(shape=VTL))+
  geom_line()+
  ggtitle("Range of motion for PC2 scores")+
  theme_mb()+
  guides(color = guide_legend(reverse=T))
ggsave("plots/PC2_performance_range.pdf", units="in", width=6.5,height=4,bg = "transparent")

#I want the median from each condition 
#this is more stable estimate of skill as compared to range

pca_skillz<-pca_dat %>%
  #get medians by condition
  mutate(ID=as.factor(ID)) %>%
  group_by(ID,Sex, VTL) %>% #order participants by Sev
  summarise(PC1_median=median(PC1),
            PC2_median=median(PC2),
            PC3_median=median(PC3),
            PC4_median=median(PC4),
            PC5_median=median(PC5),
            
            PC1_min=min(PC1),
            PC2_min=min(PC2),
            PC3_min=min(PC3),
            PC4_min=min(PC4),
            PC5_min=min(PC5),
            
            PC1_max=max(PC1),
            PC2_max=max(PC2),
            PC3_max=max(PC3),
            PC4_max=max(PC4),
            PC5_max=max(PC5)) %>%
  
  #calculate long-short difference [PC2] as a metric of skill
  mutate(VTL=as.factor(VTL))%>%
  pivot_wider(names_from=VTL, values_from=c(PC1_median,PC2_median,PC3_median,PC4_median,PC5_median,PC1_min,PC2_min,PC3_min,PC4_min,PC5_min,PC1_max,PC2_max,PC3_max,PC4_max,PC5_max)) %>%
  mutate(VTL_skill=PC2_median_Short-PC2_median_Long)


pca_skillz<-left_join(pca_skillz, demographics, by=c("ID","Sex"))
pca_skillz<- ungroup(pca_skillz) %>% #ungroup so I can fiddle factors
  arrange(Sex,Group,VTL_skill) 
pca_skillz$ID<- fct_inorder(pca_skillz$ID) #reorder factors by first appearance

#add skill categories
participants_poor_F<-pca_skillz %>% 
  filter(Sex=="F") %>%
  arrange(VTL_skill) %>% #ascending by default
  dplyr::select(ID)%>%
  slice(1:5)%>%
  as.data.frame()

participants_good_F<-pca_skillz %>% 
  filter(Sex=="F") %>%
  arrange(desc(VTL_skill)) %>% #descending
  dplyr::select(ID)%>%
  slice(1:6)%>%
  as.data.frame()

participants_poor_M<-pca_skillz %>% 
  filter(Sex=="M") %>%
  arrange(VTL_skill) %>% #ascending by default
  dplyr::select(ID)%>%
  slice(1:5)%>%
  as.data.frame()

participants_good_M<-pca_skillz %>% 
  filter(Sex=="M") %>%
  arrange(desc(VTL_skill)) %>% #descending
  dplyr::select(ID)%>%
  slice(1:5) %>%
  as.data.frame()

pca_skillz$Skill_class<-"Moderate"
pca_skillz<-pca_skillz%>% 
  mutate(Skill_class=replace(Skill_class,as.character(ID) %in% as.character(participants_good_F$ID), "Good")) %>%
  mutate(Skill_class=replace(Skill_class,as.character(ID) %in% as.character(participants_good_M$ID), "Good")) %>%
  mutate(Skill_class=replace(Skill_class,as.character(ID) %in% as.character(participants_poor_F$ID), "Poor")) %>%
  mutate(Skill_class=replace(Skill_class,as.character(ID) %in% as.character(participants_poor_M$ID), "Poor"))

pca_dat$Skill_class<-"Moderate"
pca_dat<-pca_dat%>% 
  mutate(Skill_class=replace(Skill_class,as.character(ID) %in% as.character(participants_good_F$ID), "Good")) %>%
  mutate(Skill_class=replace(Skill_class,as.character(ID) %in% as.character(participants_good_M$ID), "Good")) %>%
  mutate(Skill_class=replace(Skill_class,as.character(ID) %in% as.character(participants_poor_F$ID), "Poor")) %>%
  mutate(Skill_class=replace(Skill_class,as.character(ID) %in% as.character(participants_poor_M$ID), "Poor"))

#Vocal Modulation skill
#Figure 3a
pca_skillz %>%
  ggplot(aes(x=VTL_skill, y=ID, color=Sex))+
  geom_point(aes(shape=Group, color=Sex), size=3)+
  geom_point(data=subset(pca_skillz, Skill_class=="Poor" & Group=="Control"), shape=21, color="black", size=3)+ #mark the good/poor voice modulators
  geom_point(data=subset(pca_skillz, Skill_class=="Good" & Group=="Control"), shape=21, color="black", size=3)+
  geom_point(data=subset(pca_skillz, Skill_class=="Poor" & Group=="Singer"), shape=24, color="black", size=3)+
  geom_point(data=subset(pca_skillz, Skill_class=="Good" & Group=="Singer"), shape=24, color="black", size=3)+
  geom_point(data=subset(pca_skillz, ID=="P13"), shape=17, linetype=2, size=3)+           #audio data not available for this speaker not anlysed further
  geom_point(data=subset(pca_skillz, ID=="P13"), shape=4, linetype=2,color="black", size=3)+
  ggtitle("Individual Scores")+
  xlab("Voice Modulation Skill")+ #0 means no change in larynx height 
  theme_mb()+
  theme(axis.text.y = element_text(size=6),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())+
  #theme_minimal_hgrid()+
  guides(color = guide_legend(reverse=T),shape = guide_legend(reverse=T))+
  scale_y_discrete(expand = expand_scale(add = 1.5))
ggsave("plots/PC2_VTL_skill_individual_diferences.png", units="in", width=8,height=5,bg = "transparent")

#Figure 3b
#choose highest and lowest performing of both sexes
pca_skillz %>%
  ggplot(aes(x=VTL_skill,fill=Sex))+
  geom_density(alpha=0.5)+
  #geom_rug(aes(color=Sex))+
  ggtitle("Distribution of Scores")+
  xlab("Voice Modulation Skill")+ #0 means no change in larynx height 
  theme_mb()+
  facet_grid(fct_rev(Group)~.)+
  guides(fill = guide_legend(reverse=T))
ggsave("plots/PC2_VTL_skill_distributions.png", units="in", width=8,height=5)

################################
###model PC2 by sex and group###
################################
#recode for more useful labels
pca_dat<-pca_dat %>%
  mutate(Condition = paste(VTL,Pitch, sep="-")) %>%
  mutate(Condition = replace(Condition,Condition=="Normal-Normal", "Baseline"))%>%
  mutate(Pitch = replace(Pitch,Pitch=="Normal", "Baseline"))%>%
  mutate(Pitch = fct_relevel(as.factor(Pitch),"Low")) %>% 
  mutate(VTL = replace(VTL,VTL=="Normal", "Baseline"))%>%
  mutate(VTL = replace(VTL,VTL=="Short", "Small"))%>%
  mutate(VTL = replace(VTL,VTL=="Long", "Large"))%>%
  mutate(VTL = fct_relevel(as.factor(VTL),"Large"))

#relevel for useful coefficients
pca_dat<-pca_dat %>% mutate(VTL = as.factor(VTL)) %>%
  mutate(VTL = fct_relevel(VTL, "Baseline"))

AIC(lmer(PC2~1+VTL*Sex*Group+(1|ID), data=pca_dat))
AIC(lmer(PC2~1+VTL*Sex*Group+(1+VTL|ID), data=pca_dat)) #model is better with random slope of VTL

mod_groupsex<-lmer(PC2~1+VTL*Sex*Group+(1+VTL|ID), data=pca_dat)
mod_groupsex<-standardize(mod_groupsex) #Gelmanized coefs

qqnorm(residuals(mod_groupsex)) #good
plot(mod_groupsex)  #good

anova(mod_groupsex, type="III")
summary(mod_groupsex)
mod_groupsex.ci<-confint(mod_groupsex)


pca_dat %>%
  ggplot(aes(x=VTL, y=PC1, fill=Sex))+
  geom_violin(trim=TRUE,alpha=0.5, draw_quantiles=c(0.25, 0.5, 0.75))+
  facet_grid(.~Word)+
  ylab("fPC1 score")+
  theme_mb()
ggsave("plots/fPC1_violins.pdf", units="in",width=6.5,height=3.25)

pca_dat %>%
  ggplot(aes(x=VTL, y=PC2,fill=Sex))+
  geom_violin(trim=TRUE,alpha=0.5, draw_quantiles=c(0.25, 0.5, 0.75))+
  facet_grid(.~Word)+
  ylab("fPC2 score")+
  theme_mb()
ggsave("plots/fPC2_violins.pdf", units="in", width=6.5,height=3.25)
