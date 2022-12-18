## Interpret posterior results for all the fixed effects
ext_fit <- extract(fit_spatial)

param_mean <- round(apply(ext_fit$beta[1501:3000,],2,mean),4)
param_upper <- round(apply(ext_fit$beta[1501:3000,],2,quantile,c(0.975)),4)
param_lower <- round(apply(ext_fit$beta[1501:3000,],2,quantile,c(0.025)),4)
param_name <- c("surg_age","crheum_hd","hypertens","isch_hd","pulm_hd","other_hd","artery","num_drug","chalson","quintmat2",
                "quintmat3","quintmat4","quintmat5","quintsoc2","quintsoc3","quintsoc4","quintsoc5","sex","unique drug",
                "hopitalization","hospital_visit","hospital_urgent")
param_trend <- ifelse(param_mean > 0, 1, 0)
param_est <- data.frame(cbind(c(1:22),param_name,param_lower,param_mean,param_upper,param_trend))
colnames(param_est) <- c("ID","Name","Lower","Mean","Upper","Size")
write.csv(param_est,"/home/bhatnagar-lab/jfu/Jay's Project/revision_param_estr.csv")

### Make a 95%CI plot
ggplot(param_est,aes(x=Name,y=as.numeric(Mean))) + geom_errorbar(aes(ymin=as.numeric(Lower), ymax=as.numeric(Upper)), width=.1) 
+ coord_flip() + geom_point(aes(x=Name,y=as.numeric(Mean))) + ggtitle("95% Credible intervals") + xlab("Predictors") 
+ ylab("Log-odds ratio") + theme(legend.position = "none") + geom_hline(yintercept = 0, color = "red", linetype="dotted") 


### Interpret posterior results for all the spatial effects 
spatial_mean <- round(apply(ext_fit$zeta[1501:3000,],2,mean),4)
spatial_upper <- round(apply(ext_fit$zeta[1501:3000,],2,quantile,c(0.75)),4)
spatial_lower <- round(apply(ext_fit$zeta[1501:3000,],2,quantile,c(0.25)),4)
spatial_num <- c("01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17")
spatial_est <- data.frame(cbind(spatial_mean,spatial_num,spatial_upper,spatial_lower))
colnames(spatial_est) <- c("est","region_num","lower","upper")
#region_number = 1:17
write.csv(spatial_est,"spatial_estr.csv")

### When plotting regional effect, I used a public geojson file that describes the borders of 17 health care districts
new_js <- sf::read_sf("region_admin_poly.geojson")
nes_js <- new_js %>% 
  dplyr::transmute(
    Region_ID = cartodb_id, 
    LABEL = res_nm_reg
  )
nes_js$region_num <- c("09","02","11","08","04","03","07","15","16","05","09","10","09","09","14","12","09","01","17","06","13")

nes_js <- merge(nes_js,spatial_est, by="region_num", all.x = TRUE)

##### To make number identifiable, I create a plot for center regions first
nes_js_center <- nes_js[-c(1,2,8,9,10,11,12,13,14,15),]

ggplot() +
  geom_sf(data = nes_js_center, aes(fill = round(as.numeric(est),2))) + geom_sf_text(
    data = nes_js_center, aes(label = round(as.numeric(est),2)),
    fontface = "bold", check_overlap = TRUE
  ) + scale_fill_gradient2() + labs(fill='Effect Size') + xlab(" ") + ylab(" ") + theme(legend.position = "none") 

#For remote regions
nes_js_boarder <- nes_js
nes_js_boarder$est[-c(1,2,8,9,10,11,12,13,14,15)] = NA
nes_js_boarder$est2 = nes_js_boarder$est
nes_js_boarder$est2[c(9:11,13)] = NA
nes_js_boarder$region_num[-c(1,2,8,9,10,11,12,13,14,15)] = NA

ggplot() +
  geom_sf(data = nes_js_boarder, aes(fill = round(as.numeric(est),2))) + geom_sf_text(
    data = nes_js_boarder, aes(label = round(as.numeric(est2),2)),
    fontface = "bold", check_overlap = TRUE
  ) + scale_fill_gradient2() + labs(fill='Effect Size') + xlab(" ") + ylab(" ") + theme(legend.position = "none") 
