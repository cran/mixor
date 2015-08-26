\name{norcag}
\alias{norcag}
\docType{data}
\title{
Attitudes Towards Sex Data from the 1989 General Social Survey
}
\description{
Each subject provided ordinal responses on three items concerning their opinion on early teens (age 14-16) having sex before marriage (Item1), a man and a woman having sex before marriage (Item2), and a married person having sex with someone other than their spouse (Item3). Data are provided as frequencies by response pattern.

}
\usage{data(norcag)}
\format{
  A data frame with 105 observations on the following 6 variables.
  \describe{
    \item{\code{ID}}{a numeric vector indicating unique patient identifier}
    \item{\code{SexItems}}{ordinal item response coded as 1 = always wrong; 2 = almost always wrong; 3 = wrong only sometimes; 4 = not wrong}
    \item{\code{int}}{a numeric vector of ones; used in the stand-alone MIXOR program to indicate the intercept}
    \item{\code{Item2vs1}}{attitude towards premarital vs teenage sex}
    \item{\code{Item3vs1}}{attitude towards extramarital vs teenage sex}
    \item{\code{freq}}{frequency weight of the pattern}
  }
}

\source{
Agresti A. and Lang J.B. (1993) A proportional odds model with subject-specific effects for repeated ordered categorical responses, \emph{Biometrika} 80, 527-534.
}
\references{
Hedeker D. and Mermelstein R.J. (1998) A multilevel thresholds of change model for analysis of stages of change data, \emph{Multivariate Behavioral Research} 33, 427-455.
}
\examples{
library("mixor")
data("norcag")
# random intercepts model assuming proportional odds for differences in item responses
Fitted.norcag<-mixor(SexItems~Item2vs1+Item3vs1, data=norcag, id=ID, 
     weights=freq, link="logit", NQ1=20)
summary(Fitted.norcag)
# random intercepts model assuming non-proportional odds for differences in item responses
Fitted.norcag.np<-mixor(SexItems~Item2vs1+Item3vs1, data=norcag, id=ID, 
     weights=freq, link="logit", NQ1=10, KG=2)
summary(Fitted.norcag.np)
# SCALING model 
Fitted.norcag.scale<-mixor(SexItems~Item2vs1+Item3vs1, data=norcag, id=ID, 
     weights=freq, link="logit", NQ1=10, KS=2)
summary(Fitted.norcag.scale)
}
\keyword{datasets}
