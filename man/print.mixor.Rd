\name{print.mixor}
\alias{print.mixor}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Printing Mixor Model Fits
}
\description{
\code{print} method for class \code{mixor}.
}
\usage{
\method{print}{mixor}(x, digits = max(3L, getOption("digits") - 3L), ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
an object of class \code{mixor}.
}
  \item{digits}{
the number of significant digits to use when printing.
}
  \item{\dots}{
further arguments passed to or from other methods.
}
}

\value{
The function \code{print.mixor} prints the model call and coefficients from the \code{mixor} object.
}
\author{
Donald Hedeker, Kellie J. Archer, Robert D. Gibbons, Rachel Nordgren 
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
See Also as \code{\link{mixord}}, \code{\link{summary.mixor}}, \code{\link{coef.mixor}}
}
\examples{
library("mixor")

data(schizophrenia)
### Random intercept
SCHIZO1.fit<-mixord(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, link="probit")
print(SCHIZO1.fit)
### Random intercept and slope	
SCHIZO2.fit<-mixord(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, which.random.slope=2, link="probit")
print(SCHIZO2.fit)
}
\keyword{ methods }
