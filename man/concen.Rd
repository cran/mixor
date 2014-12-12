\name{concen}
\alias{concen}
\docType{data}
\title{
Twins Trouble Concentrating Data
}
\description{
Monozygotic and dizygotic twins trouble concentrating data 
}
\usage{data(concen)}
\format{
  A data frame with 16 observations on the following 6 variables.
  \describe{
    \item{\code{ID}}{a numeric vector indicating the unique twin pair ID}
    \item{\code{TConcen}}{an indicator variable representing trouble concentrating (0 = absent, 1 = present)}
    \item{\code{int}}{a numeric vector of ones; used in the stand-alone MIXOR program to represent the intercept}
    \item{\code{Mz}}{an indicator variable representing MZ twins (1 = MZ, 0 = DZ)}
    \item{\code{Dz}}{an indicator variable representing DZ twins (1 = DZ, 0 = MZ)}
    \item{\code{freq}}{a numeric vector representing the frequency of the pattern}
  }
}

\references{
Ramakrishnan V. et al (1992) Elementary methods for the analysis of dichotomous outcomes in unselected samples of twins \emph{Genetic Epidemiology} 9, 273-287.
}
\examples{
library("mixor")

data(concen)
# sort the data by twin pair ID
concen<-concen[order(concen$ID),]
# common ICC probit model
Common.ICC<-mixord(TConcen~Mz, data=concen, id=ID, weights=freq, 
     link="probit", NQ1=10, random.effect.mean=FALSE)
summary(Common.ICC)
# Varying ICC probit model
Varying.ICC<-mixord(TConcen~Mz+Dz, data=concen, id=ID, weights=freq, 
    which.random.slope=1:2, exclude.fixed.effect=2, link="probit", 
    NQ1=20, random.effect.mean=FALSE, UNID=1)
summary(Varying.ICC)
}
\keyword{datasets}
