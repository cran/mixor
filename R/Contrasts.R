Contrasts <-
function(fit, contrast.matrix) {
        thresholds<-fit$Model[,"Estimate"]%*%contrast.matrix
        thresholds.se<-sqrt(diag(t(contrast.matrix)%*%fit$varcov%*%contrast.matrix))
        Z.thresh <- thresholds/thresholds.se
        P.thresh <- 2*(1-pnorm(abs(Z.thresh)))
        Thresholds<- cbind(Estimate=as.numeric(thresholds), SE=thresholds.se, Z=as.numeric(Z.thresh), P.value=as.numeric(P.thresh))
dimnames(contrast.matrix)[[1]]<-dimnames(fit$Model)[[1]]
        dimnames(contrast.matrix)[[2]]<-1:dim(contrast.matrix)[2]
        dimnames(Thresholds)[[1]]<-dimnames(contrast.matrix)[[2]]
list(contrast.matrix=contrast.matrix, Contrasts=Thresholds)
}
