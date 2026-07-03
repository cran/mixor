\name{summary.mixor}
\alias{summary.mixor}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Summarize Mixor Model Fit
}
\description{
\code{summary} method for class \code{mixor}.}
\usage{
\method{summary}{mixor}(object, digits=max(3, getOption("digits") - 2), 
    signif.stars=TRUE, dig.tst = max(1, min(5, digits - 1)), ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
an object of class \code{mixor}.
}
  \item{digits}{minimum number of significant digits to be used for most numbers.}
  \item{signif.stars}{logical; if TRUE, P-values are additionally encoded visually as "significance stars" in order to help scanning of long coefficient tables.}
  \item{dig.tst}{minimum number of significant digits for the test statistics.}
  \item{\dots}{other arguments.}
}

\value{
Prints the following items extracted from the fitted \code{mixor} object:
the call to mixor (model formula), mode Deviance, Log-likelihood, RIDGEMAX, AIC, SBC, and model paramter estimates along with their SE, Z, and p-value.
}
\author{
Kellie J. Archer, Donald Hedeker, Rachel Nordgren, Robert D. Gibbons
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
See Also \code{\link{mixor}}, \code{\link{print.mixor}}, \code{\link{coef.mixor}}
}
\examples{
library("mixor")
data("schizophrenia")
### Random intercept
SCHIZO1.fit<-mixor(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, link="probit")
summary(SCHIZO1.fit)
}
\keyword{ methods }