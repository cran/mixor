\name{Contrasts}
\alias{Contrasts}
\title{
Estimate Contrasts for Mixor Object
}
\description{
This function is used to specify and estimate contrasts of interest for a \code{mixor} object.
}
\usage{
Contrasts(fit, contrast.matrix)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{fit}{
A \code{mixor} object.
}
  \item{contrast.matrix}{
A matrix of contrasts of interest. Each column in the contrast matrix should be of the same length as the \code{coef} vector of the fitted \code{mixor} object.
}
}
\value{
A list containing the following components:
\item{contrast.matrix}{a matrix specifying the contrasts of interest.}
\item{Contrasts}{a matrix containing the estimates, SEs, Z-values, and p-values for the contrasts of interest.}
}
\author{
Donald Hedeker, Kellie J. Archer, Robert D. Gibbons, Rachel Nordgren 
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
See Also as \code{\link{mixor}}, \code{\link{coef.mixor}}, \code{\link{vcov.mixor}}
}
\examples{
library("mixor")
### students in classrooms analysis with varying Sex effect across time intervals
data(SmokeOnset)
library(survival)
students.mixord<-mixord(Surv(smkonset,censor)~SexMale+cc+tv, data=SmokeOnset, 
     id=class, link="cloglog", KG=1, NQ1=20, IADD=1)
students.mixord
### Estimate the intercepts using the Intercept and Thresholds
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
}
\keyword{ methods }