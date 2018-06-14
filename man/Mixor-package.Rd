\name{mixor-package}
\alias{mixor-package}
\docType{package}
\title{
Mixed-Effects Ordinal Regression Analysis
}
\description{
This package provides a function \code{mixor} for fitting mixed-effects ordinal and binary response models and associated methods for printing, summarizing, extracting estimated coefficients and the variance-covariance matrix, and estimating contrasts for the fitted models.
}
\details{
\packageDESCRIPTION{mixor}
\packageIndices{mixor}
This package provides a function \code{mixor} for fitting mixed-effects ordinal and binary response models and associated methods for printing, summarizing, extracting estimated coefficients and the variance-covariance matrix, and estimating contrasts for the fitted models.
}
\author{
\packageAuthor{mixor} Kellie J. Archer, Donald Hedeker, Rachel Nordgren, Robert D. Gibbons
Maintainer: \packageMaintainer{mixor} Kellie J. Archer <archer.43@osu.edu>
}
\references{
Hedeker D. and Gibbons R.D. (1996) A computer program for mixed-effects ordinal regression analysis. \emph{Computer Methods and Programs in Biomedicine} 49, 157-176.
}
\keyword{ package }
\keyword{ models }
\keyword{ regression }
\seealso{
	See also \code{\link{mixor}}, and examples in \code{\link{norcag}}, \code{\link{irt.data}}, \code{\link{concen}}, \code{\link{SmokeOnset}}, \code{\link{schizophrenia}}
}
\examples{
library("mixor")
data("SmokingPrevention")
# data frame must be sorted by id variable
SmokingPrevention<-SmokingPrevention[order(SmokingPrevention$class),]
# school model
Fitted.school<-mixor(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
   id=school, link="logit")
summary(Fitted.school)
vcov(Fitted.school)
# students in classrooms analysis 
Fitted.students<-mixor(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
   id=class, link="logit", nAGQ=11)
summary(Fitted.students)
coef(Fitted.students)
vcov(Fitted.students)
cm<-matrix(c(-1,-1,-1,0,0,0,0,0,                                             
		0,0,0,0,0,0,0,0,
		0,0,0,1,0,0,0,1),
		ncol=3,byrow=TRUE)
Contrasts(Fitted.students, contrast.matrix=cm)
}