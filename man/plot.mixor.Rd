\name{plot.mixor}
\alias{plot.mixor}
\title{
Plot empirical Bayes estimates of random effects for a Mixor Model Object
}
\description{
Plot of empirical Bayes estimates of random effects from a fitted \code{mixor} model object. 
}
\usage{
\method{plot}{mixor}(x, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
a \code{mixor} object.
}
  \item{\dots}{other arguments.}
}

\value{
For each random effect, a histogram and normal quantile-quantile plot of empirical Bayes estimates. 
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
plot(SCHIZO1.fit)
}
\keyword{ methods }
