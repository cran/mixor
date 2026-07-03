\name{SmokeOnset}
\alias{SmokeOnset}
\docType{data}
\title{
Smoking Onset Data
}
\description{
From the Television School and Family Smoking Prevention and Cessation Project, a study designed to increase knowledge of the effects of tobacco use in school-age children.
}
\usage{data(SmokeOnset)}
\format{
  A data frame with 1556 observations on the following 10 variables.
  \describe{
    \item{\code{school}}{a numeric vector identifying the unique school in the Los Angeles area}
    \item{\code{class}}{a numeric vector identifying classroom}
    \item{\code{student}}{a numeric vector identifying student}
    \item{\code{smkonset}}{a numeric vector indicating time to smoking or right-censoring (post-intervention=1, 1 year follow-up=2, and 2 year follow-up=3). For these data right-censoring means that the observation was observed at the indicated timepoint (and had not smoked at that timepoint or at earlier timepoints), but was not observed at subsequent timepoints or thereafter.}
    \item{\code{event}}{a numeric vector indicating whether the student smoked (1) or did not (0)}
    \item{\code{int}}{a numeric vector of ones; used in stand-alone MIXOR program to represent the intercept}
    \item{\code{SexMale}}{a numeric vector indicating gender of the student (0=Female, 1=Male)}
    \item{\code{cc}}{a numeric vector indicating whether the school was randomized to a social-resistance classroom curriculum (1=Yes, 0=No)}
    \item{\code{tv}}{a numeric vector indicating whether the school was randomized to a media (television) intervention (1=Yes, 0=No)}
    \item{\code{cctv}}{a numeric vector indicating whether the school was randomized to CC combined with TV}
  }
}
\references{
Flay B.R., Brannon B.R., Johnson C.A. et al. (1989) The television, school and family smoking cessation and prevention project: I. Theoretical basis and program development, \emph{Preventive Medicine} 17, 585-607.

Hedeker D., Siddiqui O. and Hu F.B. Random-effects regression analysis of correlated grouped-time survival data, \emph{Statistical Methods in Medical Research} 9,161-179.

Hedeker D and Gibbons R.D. (2006) \emph{Longitudinal Data Analysis}, Wiley, Hoboken, New Jesery.
}
\examples{
library("mixor")
data("SmokeOnset")
require("survival")
### Grouped survival time example
### students in classrooms analysis 
Surv.mixor<-mixor(Surv(smkonset,event)~SexMale+cc+tv, data=SmokeOnset, 
     id=class, link="cloglog", nAGQ=20, IADD=1)
Surv.mixor
vcov(Surv.mixor)
### students in schools analysis 
School.mixor<-mixor(Surv(smkonset,event)~SexMale+cc+tv, data=SmokeOnset, 
     id=school, link="cloglog", nAGQ=20, IADD=1)
School.mixor
vcov(School.mixor)
### students in classrooms analysis with varying Sex effect across time intervals
students.mixor<-mixor(Surv(smkonset,event)~SexMale+cc+tv, data=SmokeOnset, 
     id=class, link="cloglog", KG=1, nAGQ=20, IADD=1)
students.mixor
}
\keyword{datasets}
