\name{coef.mixor}
\alias{coef.mixor}
\title{
Extract Model Coefficients
}
\description{
\code{coef.mixor} is a generic function which extracts the model coefficients from a fitted model object fit using \code{mixord}
}
\usage{
\method{coef}{mixor}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
a \code{mixor} object.
}
  \item{\dots}{other arguments.}
}

\value{
Coefficients extracted from the model object.
}

\author{
Donald Hedeker, Kellie J. Archer, Robert D. Gibbons, Rachel Nordgren
}


\seealso{
See Also as \code{\link{mixord}}, \code{\link{summary.mixor}}
}
\examples{
library("mixor")
data(schizophrenia)
### Random intercept model
SCHIZO1.fit<-mixord(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
   id=id, link="probit")
coef(SCHIZO1.fit)
}

\keyword{ methods }
