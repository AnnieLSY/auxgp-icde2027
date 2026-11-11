args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) stop("need model X_train y_train X_test output seed")
model <- args[1]
Xtr <- as.matrix(read.csv(args[2], header = FALSE))
y <- as.numeric(read.csv(args[3], header = FALSE)[, 1])
Xte <- as.matrix(read.csv(args[4], header = FALSE))
set.seed(as.integer(args[6]))

if (model == "gblup") {
  suppressMessages(library(rrBLUP))
  fit <- mixed.solve(y = y, Z = Xtr)
  prediction <- as.numeric(fit$beta[1] + Xte %*% fit$u)
} else if (model == "bayesb") {
  suppressMessages(library(BGLR))
  prefix <- paste0(dirname(args[5]), "/bglr_", Sys.getpid(), "_")
  fit <- BGLR(y = y, ETA = list(list(X = Xtr, model = "BayesB")),
              nIter = 12000, burnIn = 2000, thin = 10,
              saveAt = prefix, verbose = FALSE)
  prediction <- as.numeric(fit$mu + Xte %*% fit$ETA[[1]]$b)
  unlink(paste0(prefix, "*"))
} else {
  stop(paste("unsupported R baseline", model))
}
write.table(prediction, file = args[5], row.names = FALSE, col.names = FALSE, sep = ",")
