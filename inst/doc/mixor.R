### R code from vignette source 'mixor.Rnw'

###################################################
### code chunk number 1: mixor.Rnw:160-161
###################################################
options(width=70)


###################################################
### code chunk number 2: mixor.Rnw:163-167
###################################################
library("mixor")
data(schizophrenia)
SCHIZO1.fit<-mixord(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, link="logit")


###################################################
### code chunk number 3: mixor.Rnw:172-173
###################################################
summary(SCHIZO1.fit)


###################################################
### code chunk number 4: mixor.Rnw:176-179
###################################################
print(SCHIZO1.fit)
coef(SCHIZO1.fit)
vcov(SCHIZO1.fit)


###################################################
### code chunk number 5: mixor.Rnw:183-187
###################################################
cm<-matrix(c(1,0,0,0,0, 0,0,
             1,0,0,0,0,-1,0, 
             1,0,0,0,0, 0,-1),ncol=3)
Contrasts(SCHIZO1.fit, contrast.matrix=cm)


###################################################
### code chunk number 6: mixor.Rnw:192-193
###################################################
plot(SCHIZO1.fit)


###################################################
### code chunk number 7: mixor.Rnw:201-204
###################################################
SCHIZO2.fit<-mixord(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, which.random.slope=2, link="logit")
summary(SCHIZO2.fit)


###################################################
### code chunk number 8: mixor.Rnw:207-208
###################################################
names(SCHIZO2.fit)


###################################################
### code chunk number 9: mixor.Rnw:211-213
###################################################
SCHIZO2.fit$RLOGL
SCHIZO1.fit$RLOGL


###################################################
### code chunk number 10: mixor.Rnw:217-220
###################################################
SCHIZO3.fit<-mixord(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, which.random.slope=2, indep.re=TRUE, link="logit", NQ1=11)
summary(SCHIZO3.fit)


###################################################
### code chunk number 11: mixor.Rnw:227-232
###################################################
data(SmokeOnset)
require(survival)
Surv.mixord<-mixord(Surv(smkonset,censor)~SexMale+cc+tv, data=SmokeOnset, 
     id=class, link="cloglog", NQ1=20, IADD=1)
summary(Surv.mixord)


###################################################
### code chunk number 12: mixor.Rnw:236-239
###################################################
School.mixord<-mixord(Surv(smkonset,censor)~SexMale+cc+tv, data=SmokeOnset, 
     id=school, link="cloglog", NQ1=20, IADD=1)
summary(School.mixord)


###################################################
### code chunk number 13: mixor.Rnw:244-247
###################################################
students.mixord<-mixord(Surv(smkonset,censor)~SexMale+cc+tv, data=SmokeOnset, 
     id=class, link="cloglog", KG=1, NQ1=20, IADD=1)
students.mixord


###################################################
### code chunk number 14: mixor.Rnw:250-260
###################################################
cm<-matrix(c(1,1,0,0,
             0,0,1,1,
             0,0,0,0,
             0,0,0,0,
             0,0,0,0,
             1,0,0,0,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1),byrow=TRUE,ncol=4)
Contrasts(students.mixord, contrast.matrix=cm)


###################################################
### code chunk number 15: mixor.Rnw:266-270
###################################################
data("norcag")
Fitted.norcag<-mixord(SexItems~Item2vs1+Item3vs1, data=norcag, id=ID, 
     weights=freq, link="logit", NQ1=20)
summary(Fitted.norcag)


###################################################
### code chunk number 16: mixor.Rnw:275-278
###################################################
Fitted.norcag.np<-mixord(SexItems~Item2vs1+Item3vs1, data=norcag, id=ID, 
     weights=freq, link="logit", NQ1=10, KG=2)
summary(Fitted.norcag.np)


###################################################
### code chunk number 17: mixor.Rnw:285-288
###################################################
Fitted.norcag.scale<-mixord(SexItems~Item2vs1+Item3vs1, data=norcag, id=ID, 
     weights=freq, link="logit", NQ1=10, KS=2)
summary(Fitted.norcag.scale)


###################################################
### code chunk number 18: mixor.Rnw:294-296
###################################################
data(concen)
concen<-concen[order(concen$ID),] # sort the data by twin pair ID


###################################################
### code chunk number 19: mixor.Rnw:299-302
###################################################
Common.ICC<-mixord(TConcen~Mz, data=concen, id=ID, weights=freq, 
     link="probit", NQ1=10, random.effect.mean=FALSE)
summary(Common.ICC)


###################################################
### code chunk number 20: mixor.Rnw:306-310
###################################################
Varying.ICC<-mixord(TConcen~Mz+Dz, data=concen, id=ID, weights=freq, 
    which.random.slope=1:2, exclude.fixed.effect=2, link="probit", 
    NQ1=20, random.effect.mean=FALSE, UNID=1)
summary(Varying.ICC)


