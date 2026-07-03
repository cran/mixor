\name{mixor}
\alias{mixor}
\title{
Mixed-Effects Ordinal Regression Analysis
}
\description{
This function fits mixed-effects ordinal and binary response models. Various link functions are supported including probit, logistic, and complementary log-log. For longitudinal data, the \code{mixor} function allows for individual varying intercepts and slopes across times.
}
\usage{
mixor(formula, data, id, which.random.slope = NA, subset, weights, 
	exclude.fixed.effect = NA, CONV = 1e-04, empirical.prior = FALSE, 
	quadrature.dist = "Normal", nAGQ = 11, adaptive.quadrature = TRUE, 
	link = "probit", KG = 0, KS = 0, IADD = -1, indep.re = FALSE, 
	random.effect.mean = TRUE, UNID = 0, vcov = TRUE)
}

\arguments{
  \item{formula}{
an object of class "\code{\link{formula}}" (or one that can be coerced to that class): a symbolic description of the model to be fitted.
}
  \item{data}{
an optional \code{data.frame} containing the variables in the model. Note the \code{data.frame} must be sorted by id prior to invoking the \code{mixor} function.
}
  \item{id}{
name of clustering variable in the \code{data.frame}
}
  \item{which.random.slope}{
\code{which.random.slope=NA} means no random slope, only a random intercept is fit. To specify a random slope, \code{which.random.slope} is the  number of the predictor variable on the RHS of the model formula. More than one variable can have a random slope.
}
  \item{subset}{
an optional vector specifying a subset of observations to be used in the fitting process.
}
  \item{weights}{
when supplied, indicates differential weights are used; otherwise each id is weighted equally. \code{weights} are the frequency weights to be assigned to each id.}
  \item{exclude.fixed.effect}{
the RHS of the model includes both fixed and random effects. Variables are identified as random effects using \code{which.random.slope}. If it is desired that a variable be included in the model without a fixed effect (random only), the number of that variable(s) in the RHS formula should be passed to \code{exclude.fixed.effects}.
}
  \item{CONV}{
convergence criteria, default is 0.0001.
}
  \item{empirical.prior}{
logical. If TRUE, uses empirical prior for random effects.
}
  \item{quadrature.dist}{
distribution for random effects. Either \code{"Normal"} (default) or \code{"Uniform"}.
}
  \item{nAGQ}{
number of quadrature nodes per dimension. For one random effect, nAGQ=20 is reasonable but should be reduced to 5-10 for multiple random effects.
}
  \item{adaptive.quadrature}{
logical indicating whether adaptive quadrature (default) or non-adaptive quadrature is performed.
}
  \item{link}{
Link function, either \code{"probit"}, \code{"logit"}, or \code{"cloglog"}
}
  \item{KG}{
an optional integer reflecting the number of covariates to interact with threshold parameters (the first KG variables on the RHS of the model formula, so the order of the variables on the RHS of the model formula is important when using \code{KG}). Note that if KG is non-zero, KS must be zero.
}
  \item{KS}{
an optional integer reflecting the number of covariates used in scaling (the first KS variables on the RHS of the model formula, so the order of the variables on the RHS of the model formula is important when using \code{KS}). Note that if KS is non-zero, KG must be zero.
}
  \item{IADD}{
indicates how XB part of model is included; -1 subtract covariates and mean of random effects from thresholds; 1 add covariates and mean of random effects to thresholds. Default is -1 (subtract XB).
}
  \item{indep.re}{
logical. If TRUE, independent random effects. Default is FALSE which assumes correlated random effects.
}
  \item{random.effect.mean}{
logical. If TRUE, estimate the mean of the random effects.
}
  \item{UNID}{
indicator variable where 0 = random effects are multi-dimensional; 1 = random effects are variables related to a uni-dimensional random effect (e.g., item indicators of a latent variable).
}
  \item{vcov}{
logical. Default is \code{vcov=TRUE} which returns estimates of the the random-effect variance-covariance matrix.  When \code{vcov=FALSE}, then estimates of the Cholesky (matrix square-root) of the random-effect variance-covariance matrix are returned.   
}
}
\details{
A typical model has the form response ~ terms where response is the (numeric) response vector and terms is a series of terms which specifies a linear predictor for response. The response can also be a censored survival response, of the form Surv(time,censorvariable)}
\value{
A list with components:
  \item{call}{Model call formula}
  \item{Deviance }{Model deviance}
  \item{Quadrature.points}{number of quadrature nodes per dimension used in fit}
  \item{Model}{a matrix summarizing the model fit including columns Estimate, SE, Z, and P-value}
  \item{varcov}{The variance-covariance matrix}
  \item{EBmean}{empirical Bayes estimates of the means of the posterior distribution for each id}
  \item{EBvar}{empirical Bayes estimates of posterior variance/covariance for each id)}
  \item{RIDGEMAX}{Ridge}
  \item{RLOGL}{Log-likelihood}
  \item{SE}{a matrix corresponding the standard errors of the parameters estimates in \code{coef} }
  \item{AIC}{AIC}
  \item{SBC}{Schwarz criterion}
  \item{AICD}{Akaike information criterion}
  \item{SBCD}{deviance information criterion}
  \item{MU}{mean of random effects}
  \item{ALPHA}{fixed effects estimates}
  \item{SIGMA}{random effects variance-covariance matrix}
  \item{GAM}{thresholds}
  \item{TAU}{scaling parameter estimates when proportional odds are not assumed. Returns when \code{KG} is non-zero.}
  \item{IADD}{indicates how the XB part of model was included; -1 subtract covariates and mean of random effects from thresholds; 1 add covariates and mean of random effects to thresholds.}
  \item{Y}{the response from the fitted model.}
  \item{X}{the random effects terms from the fitted model.}
  \item{W}{the fixed effects terms from the fitted model.}
  \item{MAXJ}{the number of ordinal levels.}
  \item{random.effect.mean}{If TRUE, the mean of the random effects was estimated.}
  \item{KS}{integer reflecting the number of covariates that were specified to use in scaling (the first KS variables on the RHS of the model formula).}
  \item{KG}{integer reflecting the number of covariates that were specified to interact with the threshold parameters (the first KG variables on the RHS of the model formula).}
  \item{id}{name of clustering variable in the \code{data.frame}}
  \item{which.random.slope}{the  number of the predictor variable on the RHS of the model formula having a random slope. When \code{which.random.slope=NA}, only a random intercept model was fit. }
  \item{ICEN}{if 1 indicates right-censoring was considered in the model fit.}
  \item{link}{Link function, either \code{"probit"}, \code{"logit"}, or \code{"cloglog"}}
  \item{terms}{terms from the fitted model.}
}
\references{
Hedeker D. and Gibbons R.D. (1996) A computer program for mixed-effects ordinal regression analysis. \emph{Computer Methods and Programs in Biomedicine} 49, 157-176.
Hedeker D and Gibbons R.D. (2006) \emph{Longitudinal Data Analysis}, Wiley, Hoboken, New Jesery.
}
\author{
Kellie J. Archer, Donald Hedeker, Rachel Nordgren, Robert D. Gibbons
}
\note{
This function is essentially a wrapper for the MIXOR stand-alone program by Hedeker and Gibbons with added generic methods in R to enhance functionality. Note that the Fortran wrapper requires missing values to be denoted by -9 rather than NA.
}

\seealso{
See Also as \code{\link{summary.mixor}}, \code{\link{predict.mixor}}, \code{\link{coef.mixor}}, \code{\link{vcov.mixor}}, \code{\link{print.mixor}}
}
\examples{
library("mixor")
data("SmokingPrevention")
# data frame must be sorted by id variable
SmokingPrevention<-SmokingPrevention[order(SmokingPrevention$class),]
# school model
Fitted.school<-mixor(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
     id=school, link="logit")
summary(Fitted.school)
vcov(Fitted.school)
# students in classrooms analysis 
Fitted.students<-mixor(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
     id=class, link="logit")
summary(Fitted.students)
coef(Fitted.students)
vcov(Fitted.students)
cm<-matrix(c(-1,-1,-1,0,0,0,0,0,
              0, 0, 0,0,0,0,0,0,
			  0, 0, 0,1,0,0,0,1),ncol=3,byrow=TRUE)
Contrasts(Fitted.students, contrast.matrix=cm)
}
\keyword{ models }
\keyword{ regression }
