args <- commandArgs(trailingOnly=TRUE)
if(length(args)<6) stop("need X y out nIter burnIn thin")
suppressMessages(library(BGLR))
X <- as.matrix(read.csv(args[1],header=FALSE))
y <- as.numeric(read.csv(args[2],header=FALSE)[,1])
set.seed(42)
prefix <- paste0(dirname(args[3]),"/bglr_",Sys.getpid(),"_")
fit <- BGLR(
  y=y,
  ETA=list(list(X=X,model="BayesB")),
  nIter=as.integer(args[4]),
  burnIn=as.integer(args[5]),
  thin=as.integer(args[6]),
  saveAt=prefix,
  verbose=FALSE
)
write.table(fit$ETA[[1]]$b,file=args[3],row.names=FALSE,col.names=FALSE,sep=",")
unlink(paste0(prefix,"*"))
