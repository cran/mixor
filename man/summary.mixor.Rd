\name{summary.mixor}
\alias{summary.mixor}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Summarize Mixor Model Fit
}
\description{
\code{summary} method for class \code{mixor}.}
\usage{
\method{summary}{mixor}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
an object of class \code{mixor}.
}
  \item{\dots}{other arguments.}
}

\value{
Prints the following items extracted from the fitted \code{mixor} object:
the call to mixor (model formula), mode Deviance, Log-likelihood, RIDGEMAX, AIC, SBC, and model paramter estimates along with their SE, Z, and p-value.
}
\author{
Donald Hedeker, Kellie J. Archer, Robert D. Gibbons, Rachel Nordgren
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
See Also as \code{\link{mixord}}, \code{\link{print.mixor}}, \code{\link{coef.mixor}}
}
\examples{
library("mixor")

data(schizophrenia)
### Random intercept
SCHIZO1.fit<-mixord(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, link="probit")
summary(SCHIZO1.fit)
}
\keyword{ methods }