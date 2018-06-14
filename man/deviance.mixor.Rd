\name{deviance.mixor}
\alias{deviance.mixor}
\title{
Return the deviance for a Fitted Mixor Model
}
\description{
Returns the deviance of a fitted \code{mixor} model. 
}
\usage{
\method{deviance}{mixor}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
a \code{mixor} object.
}
  \item{\dots}{other arguments.}
}

\value{
Deviance of the fitted \code{mixor} model.
}

\author{
Kellie J. Archer, Donald Hedeker, Rachel Nordgren, Robert D. Gibbons
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
See Also as \code{\link{mixor}}, \code{\link{summary.mixor}}, \code{\link{logLik.mixor}}
}
\examples{
library("mixor")
data("schizophrenia")
### Random intercept
SCHIZO1.fit<-mixor(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, link="probit")
summary(SCHIZO1.fit)
deviance(SCHIZO1.fit)
}
\keyword{ methods }
