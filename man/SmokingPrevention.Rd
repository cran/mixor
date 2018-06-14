\name{SmokingPrevention}
\alias{SmokingPrevention}
\docType{data}
\title{
Television School and Family Smoking Prevention and Cessation Project Data
}
\description{
From the Television School and Family Smoking Prevention and Cessation Project, a study designed to increase knowledge of the effects of tobacco use in school-age children.
}
\usage{data(SmokingPrevention)}
\format{
  A data frame with 1600 observations on the following 9 variables.
  \describe{
    \item{\code{school}}{a numeric vector identifying the Los Angeles school}
    \item{\code{class}}{a numeric vector identifying classroom}
    \item{\code{thksord}}{a numeric vector reflecting quartiles of the post-intervention tobacco and health knowledge scale scor}
    \item{\code{thksbin}}{a binary version of the post-intervention tobacco and health knowledge scale score}
    \item{\code{int}}{a numeric vector of ones; used in the stand-alone MIXOR program to represent the intercept}
    \item{\code{thkspre}}{the pre-intervention tobacco and health knowledge scale score}
    \item{\code{cc}}{a numeric vector indicating whether the school was randomized to a social-resistance classroom curriculum (1=Yes, 0=No}
    \item{\code{tv}}{a numeric vector indicating whether the school was randomized to a media (television) intervention (1=Yes, 0=No)}
    \item{\code{cctv}}{a numeric vector indicating whether the school was randomized to CC combined with TV}
  }
}
\references{
Flay B.R., Brannon B.R., Johnson C.A. et al. (1988) The television, school and family smoking cessation and prevention project: I. Theoretical basis and program development, \emph{Preventive Medicine} 17, 585-607.

Hedeker D. and Gibbons R.D. (1996) A computer program for mixed-effects ordinal regression analysis. \emph{Computer Methods and Programs in Biomedicine} 49, 157-176.

Hedeker D and Gibbons R.D. (2006) \emph{Longitudinal Data Analysis}, Wiley, Hoboken, New Jesery.
}
\examples{
library("mixor")
data("SmokingPrevention")
# sort dataset by id; note for these data sorting by class will also sort by school
SmokingPrevention<-SmokingPrevention[order(SmokingPrevention$class),]
# school model
Fitted.school<-mixor(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
     id=school, link="logit")
summary(Fitted.school)
vcov(Fitted.school)
# students in classrooms analysis; not run 
#Fitted.students<-mixor(thksord~thkspre+cc+tv+cctv, data=SmokingPrevention, 
#     id=class, link="logit")
#summary(Fitted.students)
#vcov(Fitted.students)

}
\keyword{datasets}
