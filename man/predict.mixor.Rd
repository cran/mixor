\name{predict.mixor}
\alias{predict.mixor}
\title{
Predict Outcome From Mixor Fitted Model
}
\description{
\code{predict} method for class \code{mixor}. 
}
\usage{
\method{predict}{mixor}(object, newdata = NULL, na.action = na.fail, ...)
}
\arguments{
  \item{object}{
an object of class \code{mixor}.
}
  \item{newdata}{
an optional data frame to be used for obtaining predictions when the random effects are zero. When \code{newdata} is not supplied, the random effects estimates are used in obtaining model predictions. All variables used in the \code{mixor} model, the fixed and the random effects models, as well as the grouping factors, must be present in the data frame. 
}
  \item{na.action}{
a function that indicates what should happen when \code{newdata} contains \code{NAS}. The default action (\code{na.fail}) causes the function to print an error message and terminate if there are any incomplete observations.
}
  \item{\dots}{
other arguments.
}
}
\details{
This function returns fitted probabilities and the predicted class from a fitted \code{mixor} object. When the \code{newdata} parameter is specified, the random effects are taken to be zero and predictions are returned. Otherwise conditional predictions, which includes both fixed and random effects terms, for the observed data are returned.
}
\value{
\item{predicted}{class specific probabilities from the fitted model.}
\item{class}{predicted class, that having the largest fitted probability, from the fitted model.}
}
\author{
Kellie J. Archer, Donald Hedeker, Rachel Nordgren, Robert D. Gibbons
}
\seealso{
See Also \code{\link{mixor}}, \code{\link{print.mixor}}
}
\examples{
library("mixor")
data("schizophrenia")
### Random intercept
SCHIZO1.fit<-mixor(imps79o ~ TxDrug + SqrtWeek + TxSWeek, data=schizophrenia, 
     id=id, link="probit")
pihat<-predict(SCHIZO1.fit)
table(pihat$class, schizophrenia$imps79o)
head(pihat$predicted)
}
\keyword{ methods }
