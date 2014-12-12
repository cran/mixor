summary.mixor <-
function(object, ...) {
    cat("\nCall:\n", paste(deparse(object$call), sep = "\n", collapse = "\n"), 
        "\n\n", sep = "")
cat("Deviance =        ", object$Deviance, "\n")
cat("Log-likelihood = ", object$RLOGL,"\n")
cat("RIDGEMAX =        ", object$RIDGEMAX,"\n")
cat("AIC =            ", object$AIC, "\n")
cat("SBC =            ", object$SBC, "\n\n")
print(object$Model)
}
