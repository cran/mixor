\name{Contrasts}
\alias{Contrasts}
\title{
Estimate Contrasts for Mixor Object
}
\description{
This function is used to specify and estimate contrasts of interest for a \code{mixor} object.
}
\usage{
Contrasts(fit, contrast.matrix, digits=max(3, getOption("digits") - 2), 
   signif.stars=TRUE, dig.tst = max(1, min(5, digits - 1)))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{fit}{
A \code{mixor} object.
}
  \item{contrast.matrix}{
A matrix of contrasts of interest. Each column in the contrast matrix should be of the same length as the \code{coef} vector of the fitted \code{mixor} object.
}
  \item{digits}{minimum number of significant digits to be used for most numbers.}
  \item{signif.stars}{logical; if TRUE, P-values are additionally encoded visually as "significance stars" in order to help scanning of long coefficient tables.}
  \item{dig.tst}{minimum number of significant digits for the test statistics.}
}
\value{
A list containing the following components:
\item{contrast.matrix}{a matrix specifying the contrasts of interest.}
\item{Contrasts}{a matrix containing the estimates, SEs, Z-values, and p-values for the contrasts of interest.}
}
\author{
Kellie J. Archer, Donald Hedeker, Rachel Nordgren, Robert D. Gibbons
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
See Also as \code{\link{mixor}}, \code{\link{coef.mixor}}, \code{\link{vcov.mixor}}
}
\examples{
library("mixor")
data("SmokingPrevention")
# data frame must be sorted by id variable
SmokingPrevention<-SmokingPrevention[order(SmokingPrevention$class),]
Fitted.students<-mixor(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
     id=class, link="logit")
summary(Fitted.students)
coef(Fitted.students)
vcov(Fitted.students)
cm<-matrix(c(-1,-1,-1,0,0,0,0,0,
              0, 0, 0,0,0,0,0,0,
			  0, 0, 0,1,0,0,0,1),ncol=3,byrow=TRUE)
Contrasts(Fitted.students, contrast.matrix=cm)
}
\keyword{ methods }