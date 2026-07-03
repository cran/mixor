\name{logLik.mixor}
\alias{logLik.mixor}
\title{
Return the log-likelihood for a Fitted Mixor Model
}
\description{
Returns the log-likelihood of a fitted \code{mixor} model. 
}
\usage{
\method{logLik}{mixor}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
a \code{mixor} object.
}
  \item{\dots}{other arguments.}
}

\value{
Log-likelihood of the fitted \code{mixor} model.
}

\author{
Kellie J. Archer, Donald Hedeker, Rachel Nordgren, Robert D. Gibbons
}
\seealso{
See Also as \code{\link{mixor}}, \code{\link{summary.mixor}}, \code{\link{deviance.mixor}}
}
\examples{
library("mixor")
data("schizophrenia")
### Random intercept
SCHIZO1.fit<-mixor(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, link="probit")
summary(SCHIZO1.fit)
logLik(SCHIZO1.fit)
}
\keyword{ methods }
