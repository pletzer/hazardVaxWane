
erf <- function(x) {
    return(2*pnorm(x*sqrt(2)) - 1.)
}

lambda_func <- function(t, param) {
    lam0 <- param[1]
    b <- param[2]
    Te <- param[3]
    sige <- param[4]
    a <- param[5]
    tau <- param[6]
    tm <- t - Te
    return (lam0*(1 + b*exp(-(t-Te)^2/(2*sige^2)))*(1 + a*exp(-t/tau)))
}

Lambda_func_primitive <- function(t, param) {
    lam0 <- param[1]
    b <- param[2]
    Te <- param[3]
    sige <- param[4]
    a <- param[5]
    tau <- param[6]

    tm <- t - Te
    num1 <- sige^2 + tau*tm
    den1 <- sqrt(2) * sige * tau
    res <- lam0*( t - a*exp(-t/tau)*tau + a*b*exp((sige^2 - 2*tau*Te)/(2*tau^2)) * sqrt(pi/2) * sige *erf(num1/den1) + b*sqrt(pi/2)* sige * erf(tm/(sqrt(2)*sige))  )
    return(res)
}

Lambda_func <- function(t, param) {
    # Make sure the integrated risk is zero at time t = 0
    return(Lambda_func_primitive(t, param) - Lambda_func_primitive(0, param))
}


test <- function() {
    param <- c(1., 2., 20, 10, -0.8, 30)
    print(sprintf("lambda = %f", lambda_func(10, param)))
    print(Lambda_func(0, param))

    tvals <- seq(0, 100, by = 1)
    lvals <- lambda_func(tvals, param)
    biglvals <- Lambda_func(tvals, param)
    png('lambda.png')
    plot(tvals, lvals)
    dev.off()

    png('biglambda.png')
    plot(tvals, biglvals)
    dev.off()
}


