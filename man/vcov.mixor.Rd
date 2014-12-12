\name{vcov.mixor}
\alias{vcov.mixor}
\title{
Return Variance-Covariance Matrix for a Mixor Model Object
}
\description{
Returns the variance-covariance matrix of a fitted \code{mixor} model object. 
}
\usage{
\method{vcov}{mixor}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
a \code{mixor} object.
}
  \item{\dots}{other arguments.}
}

\value{
A matrix of the estimated covariances between the parameter estimates in the model.
}

\author{
Donald Hedeker, Kellie J. Archer, Robert D. Gibbons, Rachel Nordgren
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
See Also as \code{\link{mixord}}, \code{\link{summary.mixor}}
}
\examples{
library("mixor")

data(schizophrenia)
### Random intercept
SCHIZO1.fit<-mixord(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, link="probit")
summary(SCHIZO1.fit)
vcov(SCHIZO1.fit)
}
\keyword{ methods }
