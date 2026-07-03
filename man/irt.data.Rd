\name{irt.data}
\alias{irt.data}
\docType{data}
\title{
Armed Servces Vocational Aptitude Battery
}
\description{
Arithmetic reasoning items from the Armed Servces Vocational Aptitude Battery
}
\usage{data(irt.data)}
\format{
  A data frame with 128 observations on the following 13 variables.
  \describe{
    \item{\code{ID}}{a numeric vector indicating unique pattern identifier}
    \item{\code{Resp}}{item response (0 = incorrect; 1 = correct)}
    \item{\code{int}}{a numeric vector of ones; used in MIXOR stand-alone program to represent the intercept}
    \item{\code{Item1}}{Arithmetic reasoning item 1 (0 = no; 1 = yes)}
    \item{\code{Item2}}{Arithmetic reasoning item 2 (0 = no; 1 = yes)}
    \item{\code{Item3}}{Arithmetic reasoning item 3 (0 = no; 1 = yes)}
    \item{\code{Item4}}{Arithmetic reasoning item 4 (0 = no; 1 = yes)}
    \item{\code{SexF}}{an indicator variable representing gender  (0 = female; 1 = male)}
    \item{\code{SexItem1}}{Gender by Item1 interaction}
    \item{\code{SexItem2}}{Gender by Item2 interaction}
    \item{\code{SexItem3}}{Gender by Item3 interaction}
    \item{\code{SexItem4}}{Gender by Item4 interaction}
    \item{\code{freq}}{frequency weight of pattern}
  }
}

\references{
Mislevy R.J. (1985) Estimation of latent group effects \emph{Journal of the American Statistical Association} 80, 993-997.
}
\examples{
library("mixor")
data("irt.data")
## Sort data by ID
irt.data<-irt.data[order(irt.data$ID),]
# 2 parameter logistic IRT model
Fitted.irt<-mixor(Resp~Item1+Item2+Item3+Item4-1, data=irt.data, id=ID, 
     weights=freq, which.random.slope=1:4, link="logit", nAGQ=10, 
	 random.effect.mean=FALSE, UNID=1, vcov=FALSE)
summary(Fitted.irt)
# 2 parameter logistic IRT model with Sex interations on item parameters
Fitted.irt2<-mixor(Resp~Item1+Item2+Item3+Item4+SexItem1+SexItem2+SexItem3+SexItem4-1, 
     data=irt.data, id=ID, weights=freq, which.random.slope=1:8, link="logit", 
	 nAGQ=10, random.effect.mean=FALSE, UNID=1, vcov=FALSE)
summary(Fitted.irt2)
}
\keyword{datasets}
