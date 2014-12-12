\name{mixor-package}
\alias{mixor-package}
\alias{mixor}
\docType{package}
\title{
Mixed-Effects Ordinal Regression Analysis
}
\description{
This package provides a function \code{mixord} for fitting mixed-effects ordinal and binary response models and associated methods for printing, summarizing, extracting estimated coefficients and the variance-covariance matrix, and estimating contrasts for the fitted models.
}
\details{
\tabular{ll}{
Package: \tab mixor\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2014-03-11\cr
License: \tab GPL2.0\cr
}
This package provides a function \code{mixord} for fitting mixed-effects ordinal and binary response models and associated methods for printing, summarizing, extracting estimated coefficients and the variance-covariance matrix, and estimating contrasts for the fitted models.
}

\author{
Donald Hedeker, Kellie J. Archer, Robert D. Gibbons, Rachel Nordgren

Maintainer: Donald Hedeker <hedeker@uic.edu>
Kellie J. Archer <kjarcher@vcu.edu>
}
\references{
Hedeker D. and Gibbons R.D. (1996) A computer program for mixed-effects ordinal regression analysis. \emph{Computer Methods and Programs in Biomedicine} 49, 157-176.
}
\keyword{ package }
\keyword{ models }
\keyword{ regression }
\seealso{
	See also \code{\link{mixord}}, and examples in \code{\link{norcag}}, \code{\link{irt.data}}, \code{\link{concen}}, \code{\link{SmokeOnset}}, \code{\link{schizophrenia}}
}
\examples{
library("mixor")

data(SmokingPrevention)
# data frame must be sorted by id variable
SmokingPrevention<-SmokingPrevention[order(SmokingPrevention$class),]
# school model
Fitted.school<-mixord(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
   id=school, link="logit")
summary(Fitted.school)
vcov(Fitted.school)
# students in classrooms analysis 
Fitted.students<-mixord(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
   id=class, link="logit", NQ1=11)
summary(Fitted.students)
coef(Fitted.students)
vcov(Fitted.students)
cm<-matrix(c(-1,-1,-1,0,0,0,0,0,                                             
		0,0,0,0,0,0,0,0,
		0,0,0,1,0,0,0,1),
		ncol=3,byrow=TRUE)
Contrasts(Fitted.students, contrast.matrix=cm)
}