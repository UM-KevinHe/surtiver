surtiver <- function(formula, data, spline="B-spline", nsplines=8, ties="Breslow", 
                  control, ...) {
  if (!ties%in%c("Breslow", "none")) stop("Invalid ties!")
  # pass ... args to surtiver.control
  extraArgs <- list(...)
  if (length(extraArgs)) {
    controlargs <- names(formals(surtiver.control))
    indx <- pmatch(names(extraArgs), controlargs, nomatch=0L)
    if (any(indx==0L))
      stop(gettextf("Argument(s) %s not matched!", 
                    names(extraArgs)[indx==0L]), domain=NA)
  }
  if (missing(control)) control <- surtiver.control(...)
  degree <- control$degree
  Terms <- terms(formula, specials=c("tv", "strata", "offset"))
  if(attr(Terms, 'response')==0) stop("Formula must have a Surv response!")
  factors <- attr(Terms, 'factors')
  terms <- row.names(factors)
  idx.r <- attr(Terms, 'response')
  idx.o <- attr(Terms, 'specials')$offset
  idx.str <- attr(Terms, 'specials')$strata
  idx.tv <- attr(Terms, 'specials')$tv
  if (is.null(idx.tv)) stop("No variable specified with time-variant effect!")
  term.ti <- terms[setdiff(1:length(terms), c(idx.r, idx.o, idx.str, idx.tv))]
  term.time <- gsub(".*\\(([^,]*),\\s+([^,]*)\\)", "\\1", terms[idx.r])
  term.event <- gsub(".*\\(([^,]*),\\s+([^,]*)\\)", "\\2", terms[idx.r])
  if (!is.null(idx.o)) term.o <- gsub(".*\\((.*)\\)", "\\1", terms[idx.o])
  term.str <- ifelse(!is.null(idx.str),
                     gsub(".*\\((.*)\\)", "\\1", terms[idx.str][1]), "strata")
  term.tv <- gsub(".*\\((.*)\\)", "\\1", terms[idx.tv])
  if (is.null(idx.str)) data[,term.str] <- "unstratified"
  data <- data[order(data[,term.str], data[,term.time], -data[,term.event]),]
  row.names(data) <- NULL
  times <- data[,term.time]; times <- unique(times); times <- times[order(times)]
  strata.noevent <- 
    unique(data[,term.str])[sapply(split(data[,term.event],data[,term.str]), 
                                   sum)==0]
  data <- data[!data[,term.str]%in%strata.noevent,] # drop strata with no event
  count.strata <- sapply(split(data[,term.str], data[,term.str]), length)
  if (any(!data[,term.event]%in%c(0,1)) |
      !is.numeric(data[,term.time]) |
      min(data[,term.time])<0) stop("Invalid Surv object!")
  # check spline-related arguments
  if (!spline%in%c("B-spline", "P-spline") | 
      is.na(suppressWarnings(as.integer(nsplines[1]))) |
      as.integer(nsplines[1])<=degree+1) 
    stop(sprintf("Invalid spline or nsplines (should be at least %.0f)!", 
                 degree+2))
  nsplines <- nsplines[1]
  # model fitting
  if (spline=="B-spline") {
    knots <- 
      quantile(data[data[,term.event]==1,term.time], 
               (1:(nsplines-degree-1))/(nsplines-degree))
    if (ties=="Breslow") {
      uniqfailtimes.str <- unname(unlist(lapply(
        split(data[data[,term.event]==1,term.time],
              data[data[,term.event]==1,term.str]), unique)))
      bases <- splines::bs(uniqfailtimes.str, degree=degree, intercept=T, 
                           knots=knots, Boundary.knots=range(times))
      if (length(term.ti)==0) {
        fit <- 
          surtiver_fixtra_bresties_fit(data[,term.event], data[,term.time], 
                                    count.strata, 
                                    as.matrix(data[,term.tv]), as.matrix(bases), 
                                    matrix(0, length(term.tv), nsplines), 
                                    matrix(0, 1), matrix(0, 1),
                                    method=control$method, 
                                    lambda=control$lambda, factor=control$factor,
                                    parallel=control$parallel,
                                    threads=control$threads, 
                                    tol=control$tol, iter_max=control$iter.max, 
                                    s=control$s, t=control$t, 
                                    btr=control$btr, stop=control$stop)
      } else {
        fit <- 
          surtiver_fixtra_bresties_fit(data[,term.event], data[,term.time],
                                    count.strata, 
                                    as.matrix(data[,term.tv]), as.matrix(bases), 
                                    matrix(0, length(term.tv), nsplines), 
                                    as.matrix(data[,term.ti]), 
                                    matrix(0,length(term.ti)),
                                    method=control$method, 
                                    lambda=control$lambda, factor=control$factor,
                                    parallel=control$parallel,
                                    threads=control$threads,
                                    tol=control$tol, iter_max=control$iter.max, 
                                    s=control$s, t=control$t, 
                                    btr=control$btr, stop=control$stop)
      }
    } else if (ties=="none") {
      bases <- 
        splines::bs(data[,term.time], degree=degree, intercept=T, 
                    knots=knots, Boundary.knots=range(times))
      if (length(term.ti)==0) {
        fit <- 
          surtiver_fixtra_fit(data[,term.event], count.strata, 
                           as.matrix(data[,term.tv]), as.matrix(bases), 
                           matrix(0, length(term.tv), nsplines), 
                           matrix(0, 1), matrix(0, 1),
                           method=control$method, 
                           lambda=control$lambda, factor=control$factor,
                           parallel=control$parallel, threads=control$threads, 
                           tol=control$tol, iter_max=control$iter.max, 
                           s=control$s, t=control$t, 
                           btr=control$btr, stop=control$stop)
      } else {
        fit <- 
          surtiver_fixtra_fit(data[,term.event], count.strata, 
                           as.matrix(data[,term.tv]), as.matrix(bases), 
                           matrix(0, length(term.tv), nsplines), 
                           as.matrix(data[,term.ti]), 
                           matrix(0, length(term.ti)),
                           method=control$method, 
                           lambda=control$lambda, factor=control$factor,
                           parallel=control$parallel, threads=control$threads, 
                           tol=control$tol, iter_max=control$iter.max, 
                           s=control$s, t=control$t, 
                           btr=control$btr, stop=control$stop)
      }
    }
    row.names(fit$ctrl.pts) <- term.tv
    fit$internal.knots <- unname(knots)
  } else if (spline=="P-spline") {
    
  }
  fit$times <- times
  fit$tvef <- splines::bs(times, degree=degree, intercept=T, knots=knots,
                          Boundary.knots=range(fit$times))%*%t(fit$ctrl.pts)
  rownames(fit$tvef) <- times
  class(fit) <- "surtiver"
  attr(fit, "spline") <- spline
  if (length(term.ti)>0) {
    fit$tief <- c(fit$tief)
    names(fit$tief) <- term.ti
  }
  colnames(fit$info) <- rownames(fit$info) <- 
    c(rep(term.tv, each=nsplines), term.ti)
  attr(fit, "nsplines") <- nsplines
  attr(fit, "degree.spline") <- degree
  attr(fit, "control") <- control
  attr(fit, "response") <- term.event
  return(fit)
}

surtiver.control <- function(tol=1e-9, iter.max=20L, method="Newton", lambda=1e8,
                          factor=10, btr="dynamic", sigma=1e-2, tau=0.6,
                          stop="incre", parallel=FALSE, threads=1L, degree=3L) {
  if (tol <= 0) stop("Invalid convergence tolerance!")
  if (iter.max <= 0 | !is.numeric(iter.max))
    stop("Invalid maximum number of iterations!")
  if (method %in% c("Newton", "ProxN")) {
    if (method=="ProxN" & (lambda <= 0 | factor < 1))
      stop("Argument lambda <=0 or factor < 1 when method = 'ProxN'!")
  } else stop("Invalid estimation method!")
  
  if (btr %in% c("none", "static", "dynamic")) {
    if (sigma <= 0 | tau <= 0) 
      stop("Search control parameter sigma <= 0 or sigma <= 0!")
    if (btr=="dynamic") {
      if (sigma > 0.5)
        stop("Search control parameter sigma > 0.5!")
      if (tau < 0.5) stop("Search control parameter tau < 0.5!")
    }
  } else stop("Invalid backtracking line search approach!")
  if (!is.logical(parallel))
    stop("Invalid argument parallel! It should be a logical constant.")
  if (!is.numeric(threads) | threads <= 0)
    stop("Invalid number of threads! It should be a positive integer.")
  if (parallel & threads==1L)
    stop(paste("Invalid number of threads for parallel computing!",
               "It should be greater than one."))
  if (!stop %in% c("incre", "ratch", "relch")) stop("Invalid stopping rule!")
  if (!is.numeric(degree) | degree < 2L) stop("Invalid degree for spline!")
  list(tol=tol, iter.max=as.integer(iter.max), method=method, lambda=lambda,
       factor=factor, btr=btr, s=sigma, t=tau, stop=stop,
       parallel=parallel, threads=as.integer(threads), degree=as.integer(degree))
}

tvef <- function(fit, times, parm) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  if (missing(times)) times <- fit$times
  if (!is.numeric(times) | min(times)<0) stop("Invalid times!")
  times <- times[order(times)]; nsplines <- attr(fit, "nsplines")
  spline <- attr(fit, "spline"); degree <- attr(fit, "degree.spline")
  knots <- fit$internal.knots; term.tv <- rownames(fit$ctrl.pts)
  if (missing(parm)) {
    parm <- term.tv
  } else if (length(parm)>0) {
    indx <- pmatch(parm, term.tv, nomatch=0L)
    if (any(indx==0L))
      stop(gettextf("%s not matched!", parm[indx==0L]), domain=NA)
  } else stop("Invalid parm!")
  if (spline=="B-spline") {
    bases <- splines::bs(times, degree=degree, intercept=T, knots=knots, 
                         Boundary.knots=range(fit$times))
    int.bases <- splines2::ibs(times, degree=degree, intercept=T, knots=knots, 
                               Boundary.knots=range(fit$times))
    ctrl.pts <- matrix(fit$ctrl.pts[term.tv%in%parm,], ncol=nsplines)
    mat.tvef <- bases%*%t(ctrl.pts); mat.cumtvef <- int.bases%*%t(ctrl.pts)
    colnames(mat.tvef) <- parm; colnames(mat.cumtvef) <- parm
    rownames(mat.tvef) <- times; rownames(mat.cumtvef) <- times
    ls <- list(tvef=mat.tvef, cumtvef=mat.cumtvef)
    return(ls)
  } else if (spline=="P-spline") {
    
  }
}

tief <- function(fit, parm) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  term.ti <- names(fit$tief)
  if (is.null(fit$tief)) return(NULL)
  if (missing(parm)) {
    parm <- term.ti
  } else if (length(parm)>0) {
    indx <- pmatch(parm, term.ti, nomatch=0L)
    if (any(indx==0L))
      stop(gettextf("%s not matched!", parm[indx==0L]), domain=NA)
    return(fit$tief[term.ti%in%parm])
  } else stop("Invalid parm!")
}

tief.zero <- function(fit, parm) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  term.ti <- names(fit$tief); term.tv <- rownames(fit$ctrl.pts)
  nsplines <- attr(fit, "nsplines")
  method <- attr(fit,"control")$method
  if (is.null(fit$tief)) return(NULL)
  if (missing(parm)) {
    parm <- term.ti
  } else if (length(parm)>0) {
    indx <- pmatch(parm, term.ti, nomatch=0L)
    if (any(indx==0L))
      stop(gettextf("%s not matched!", parm[indx==0L]), domain=NA)
  }
  rownames.info <- c(rep(term.tv, each=nsplines), term.ti)
  if (method=="Newton") {
    invinfo <- solve(fit$info)
  } else if (method=="ProxN") {
    invinfo <- solve(fit$info+diag(sqrt(.Machine$double.eps),dim(fit$info)[1]))
  }
  if (spline=="B-spline") {
    
  }
  est.ti <- fit$tief[term.ti%in%parm]
  se.ti <- c(sqrt(diag(as.matrix(invinfo[rownames.info%in%parm,
                                         rownames.info%in%parm]))))
  stat <- est.ti/se.ti
  p.value <- 2*pmin(pnorm(stat, lower.tail=F), 1-pnorm(stat, lower.tail=F))
  mat.ti <- cbind(est.ti, se.ti, stat, p.value)
  colnames(mat.ti) <- c("est", "se", "z", "p")
  rownames(mat.ti) <- parm
  return(mat.ti)
}

tvef.zero <- function(fit, times, parm) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  if (missing(times)) {
    times <- fit$times
  } else {
    if (!is.numeric(times) | min(times)<0) stop("Invalid times!")
  }
  term.ti <- names(fit$tief); term.tv <- rownames(fit$ctrl.pts)
  spline <- attr(fit, "spline"); nsplines <- attr(fit, "nsplines")
  degree <- attr(fit, "degree"); knots <- fit$internal.knots
  method <- attr(fit,"control")$method
  if (missing(parm)) {
    parm <- term.tv
  } else if (length(parm)>0) {
    indx <- pmatch(parm, term.tv, nomatch=0L)
    if (any(indx==0L))
      stop(gettextf("%s not matched!", parm[indx==0L]), domain=NA)
  }
  rownames.info <- c(rep(term.tv, each=nsplines), term.ti)
  if (method=="Newton") {
    invinfo <- solve(fit$info)
  } else if (method=="ProxN") {
    invinfo <- solve(fit$info+diag(sqrt(.Machine$double.eps),dim(fit$info)[1]))
  }
  if (spline=="B-spline") {
    bases <- splines::bs(times, degree=degree, intercept=T, knots=knots, 
                         Boundary.knots=range(fit$times))
    ctrl.pts <- matrix(fit$ctrl.pts[term.tv%in%parm,], ncol=nsplines)
    ls <- lapply(parm, function(tv) {
      est <- bases%*%ctrl.pts[parm%in%tv,]
      se <- sqrt(apply(bases, 1, function(r) {
        idx <- rownames.info%in%tv
        return(t(r)%*%invinfo[idx, idx]%*%r)}))
      stat <- est / se
      p.upper <- pnorm(stat, lower.tail=F)
      p.value <- 2*pmin(p.upper, 1-p.upper)
      mat <- cbind(est, se, stat, p.value)
      colnames(mat) <- c("est", "se", "z", "p")
      rownames(mat) <- times
      return(mat)})
    names(ls) <- parm
  } else if (spline=="P-spline") {
    
  }
  return(ls)
}

tvef.ph <- function(fit, parm) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  nsplines <- attr(fit, "nsplines"); spline <- attr(fit, "spline")
  term.ti <- names(fit$tief); term.tv <- rownames(fit$ctrl.pts)
  method <- attr(fit,"control")$method
  if (missing(parm)) {
    parm <- term.tv
  } else if (length(parm)>0) {
    indx <- pmatch(parm, term.tv, nomatch=0L)
    if (any(indx==0L))
      stop(gettextf("%s not matched!", parm[indx==0L]), domain=NA)
  } else stop("Invalid parm!")
  rownames.info <- c(rep(term.tv, each=nsplines), term.ti)
  if (method=="Newton") {
    invinfo <- solve(fit$info)
  } else if (method=="ProxN") {
    invinfo <- solve(fit$info+diag(sqrt(.Machine$double.eps),dim(fit$info)[1]))
  }
  if (spline=="B-spline") {
    mat.contrast <- diff(diag(nsplines))
    ctrl.pts <- matrix(fit$ctrl.pts[term.tv%in%parm,], ncol=nsplines)
    mat.test <- sapply(parm, function(tv) {
      bread <- mat.contrast%*%ctrl.pts[parm%in%tv,]
      idx <- rownames.info%in%tv
      meat <- solve(mat.contrast%*%invinfo[idx,idx]%*%t(mat.contrast))
      stat <- t(bread)%*%meat%*%bread
      p.value <- pchisq(stat, nsplines-1, lower.tail=F)
      return(c(stat, nsplines-1, p.value))})
    colnames(mat.test) <- parm
    rownames(mat.test) <- c("chisq", "df", "p")
    return(t(mat.test))
  } else if (spline=="P-spline") {
    
  }
}

plot.surtiver <- function(fit, times, parm, CI=TRUE, level=0.95, exponentiate=FALSE, 
                       xlab, xlim, ylim, save=FALSE, ...) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  term.event <- attr(fit, "response")
  xlab <- ifelse(missing(xlab), "time", xlab)
  missingxlim <- missing(xlim); missingylim <- missing(ylim)
  if (!is.logical(save)) stop("Invalid save!")
  if (!is.logical(exponentiate)) stop("Invalid exponentiate!")
  ls.tvef <- confint.surtiver(fit, times, parm, level)$tvef
  if (length(ls.tvef)==0) stop("No time-varying effect chosen!")
  if (!require(ggplot2)) install.packages('ggplot2')
  library(ggplot2)
  ls.plts <- lapply(names(ls.tvef), function(tv) {
    df.tv <- data.frame(ls.tvef[[tv]], as.numeric(rownames(ls.tvef[[tv]])))
    names(df.tv) <- c("est", "lower", "upper", "time")
    for (col in names(df.tv)) {
      range.tmp <- range(df.tv[!is.infinite(df.tv[,col]),col])
      df.tv[is.infinite(df.tv[,col]) & df.tv[,col] < 0, col] <- range.tmp[1]
      df.tv[is.infinite(df.tv[,col]) & df.tv[,col] > 0, col] <- range.tmp[2]
    }
    row.names(df.tv) <- NULL
    if (exponentiate) df.tv[,-4] <- exp(df.tv[,-4])
    plt <- ggplot(data=df.tv, aes(x=time)) +
      geom_hline(yintercept=ifelse(exponentiate,1,0),
                 color="black", size=0.3, linetype="dashed") +
      geom_line(aes(y=est, linetype="estimate"), size=0.9)
    if (CI) {
      plt <- plt +
        geom_ribbon(aes(ymin=lower, ymax=upper,
                        fill=paste0(round(100*level),"% CI")), alpha=0.4)
    }
    if (missingxlim) {
      plt <- plt + scale_x_continuous(name=xlab, expand=c(1,1)/20)
    } else {
      if (!is.numeric(xlim)) stop("Invalid xlim!")
      plt <- plt + scale_x_continuous(name=xlab, expand=c(1,1)/20, limits=xlim)
    }
    if (missingylim) {
      plt <- plt +
        scale_y_continuous(name=ifelse(exponentiate,"hazard ratio","coefficient"),
                           expand=c(1,1)/20)
    } else {
      if (!is.numeric(ylim)) stop("Invalid ylim!")
      plt <- plt +
        scale_y_continuous(name=ifelse(exponentiate,"hazard ratio","coefficient"),
                           expand=c(1,1)/20, limits=ylim)
    }
    plt +
      scale_linetype_manual("", values="solid") +
      scale_fill_manual("", values="grey") +
      ggtitle(paste0(tv, " (", term.event, ")")) + theme_bw() +
      theme(plot.title=element_text(hjust=0),
            panel.background=element_blank(), panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(), panel.border=element_blank(),
            axis.line=element_line(color="black"),
            axis.title=element_text(size=18, margin=margin(t=0,r=0,b=0,l=0)),
            axis.text=element_text(size=14), text=element_text(size=14),
            legend.title=element_blank(), legend.text=element_text(size=14),
            legend.position=c(0.5, 1), legend.box="horizontal")
  })
  if (save) {
    return(ls.plts)
  } else {
    return(ggpubr::ggarrange(plotlist=ls.plts, common.legend=T, ...))
  }
}

plot.surtiver <- function(fit, times, parm, CI=TRUE, level=0.95, exponentiate=FALSE, 
                          xlab, ylab, xlim, ylim, save=FALSE, allinone=FALSE, 
                          title, linetype, fill, color, labels, expand, ...) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  if (!is.logical(save)) stop("Invalid save!")
  if (!is.logical(exponentiate)) stop("Invalid exponentiate!")
  term.event <- attr(fit, "response")
  if (missing(xlab)) xlab <- "time"
  if (missing(ylab)) ylab <- ifelse(exponentiate,"hazard ratio","coefficient")
  missingxlim <- missing(xlim); missingylim <- missing(ylim); 
  missingtitle <- missing(title); missinglty <- missing(linetype)
  missingfill <- missing(fill); missingcolor <- missing(color)
  defaultcols <- c("#F8766D","#A3A500","#00BF7D","#00B0F6","#E76BF3")
  defaultltys <- c("solid", "dashed", "dotted", "dotdash", "longdash")
  if (missing(expand)) expand <- c(1,1)/100
  ls.tvef <- confint.surtiver(fit, times, parm, level)$tvef
  if (length(ls.tvef)==0) stop("No time-varying effect chosen!")
  if (missing(labels)) labels <- names(ls.tvef)
  if (!require(ggplot2)) install.packages('ggplot2')
  library(ggplot2)
  options(stringsAsFactors=F)
  if (!allinone) {
    ls.plts <- lapply(names(ls.tvef), function(tv) {
      df.tv <- data.frame(ls.tvef[[tv]], as.numeric(rownames(ls.tvef[[tv]])))
      names(df.tv) <- c("est", "lower", "upper", "time")
      for (col in names(df.tv)) {
        range.tmp <- range(df.tv[!is.infinite(df.tv[,col]),col])
        df.tv[is.infinite(df.tv[,col]) & df.tv[,col] < 0, col] <- range.tmp[1]
        df.tv[is.infinite(df.tv[,col]) & df.tv[,col] > 0, col] <- range.tmp[2]
      }
      row.names(df.tv) <- NULL
      if (exponentiate) df.tv[,-4] <- exp(df.tv[,-4])
      plt <- ggplot(data=df.tv, aes(x=time)) +
        geom_hline(yintercept=ifelse(exponentiate,1,0),
                   color="black", size=0.3, linetype="dashed") +
        geom_line(aes(y=est, linetype="estimate"), size=0.9)
      if (CI) {
        plt <- plt +
          geom_ribbon(aes(ymin=lower, ymax=upper,
                          fill=paste0(round(100*level),"% CI")), alpha=0.4)
      }
      if (missingxlim) {
        plt <- plt + scale_x_continuous(name=xlab, expand=expand)
      } else {
        if (!is.numeric(xlim)) stop("Invalid xlim!")
        plt <- plt + scale_x_continuous(name=xlab, expand=expand, limits=xlim)
      }
      if (missingylim) {
        plt <- plt +
          scale_y_continuous(name=ylab, expand=expand)
      } else {
        if (!is.numeric(ylim)) stop("Invalid ylim!")
        plt <- plt +
          scale_y_continuous(name=ylab, expand=expand, limits=ylim)
      }
      plt +
        scale_linetype_manual("", values="solid") +
        scale_fill_manual("", values="grey") +
        ggtitle(paste0(tv, " (", term.event, ")")) + theme_bw() +
        theme(plot.title=element_text(hjust=0),
              panel.background=element_blank(), panel.grid.major=element_blank(),
              panel.grid.minor=element_blank(), panel.border=element_blank(),
              axis.line=element_line(color="black"),
              axis.title=element_text(size=18, margin=margin(t=0,r=0,b=0,l=0)),
              axis.text=element_text(size=14), text=element_text(size=14),
              legend.title=element_blank(), legend.text=element_text(size=14),
              legend.position=c(0.5, 1), legend.box="horizontal")
    })
    if (save) {
      return(ls.plts)
    } else {
      return(ggpubr::ggarrange(plotlist=ls.plts, common.legend=T, ...))
    }
  } else {
    if (length(names(ls.tvef)) > 5) stop("Number of parameters greater than 5!")
    df <- do.call(rbind, lapply(names(ls.tvef), function(tv) {
      df.tv <- data.frame(ls.tvef[[tv]], as.numeric(rownames(ls.tvef[[tv]])))
      names(df.tv) <- c("est", "lower", "upper", "time")
      for (col in names(df.tv)) {
        range.tmp <- range(df.tv[!is.infinite(df.tv[,col]),col])
        df.tv[is.infinite(df.tv[,col]) & df.tv[,col] < 0, col] <- range.tmp[1]
        df.tv[is.infinite(df.tv[,col]) & df.tv[,col] > 0, col] <- range.tmp[2]
      }
      if (exponentiate) df.tv[,-4] <- exp(df.tv[,-4])
      df.tv[,"parm"] <- tv
      row.names(df.tv) <- NULL
      return(df.tv)}))
    plt <- ggplot(data=df, aes(x=time, group=parm)) +
      geom_hline(yintercept=ifelse(exponentiate,1,0),
                 color="black", size=0.3, linetype="dashed") +
      geom_line(aes(y=est, linetype=parm, color=parm), size=0.9)
    if (CI) {
      plt <- plt +
        geom_ribbon(aes(ymin=lower, ymax=upper, fill=parm), alpha=0.1)
    }
    if (missingxlim) {
      plt <- plt + scale_x_continuous(name=xlab, expand=expand)
    } else {
      if (!is.numeric(xlim)) stop("Invalid xlim!")
      plt <- plt + scale_x_continuous(name=xlab, expand=expand, limits=xlim)
    }
    if (missingylim) {
      plt <- plt +
        scale_y_continuous(name=ylab, expand=expand)
    } else {
      if (!is.numeric(ylim)) stop("Invalid ylim!")
      plt <- plt +
        scale_y_continuous(name=ylab, expand=expand, limits=ylim)
    }
    if (!missinglty) {
      plt <- plt + scale_linetype_manual(NULL, values=linetype, labels=labels)
    } else {
      plt <- plt + scale_linetype_manual(NULL, values=defaultltys[1:length(names(ls.tvef))], 
                                         labels=labels)
    }
    if (!missingcolor) {
      plt <- plt + scale_color_manual(NULL, values=color, labels=labels)
    } else {
      plt <- plt + scale_color_manual(NULL, values=defaultcols[1:length(names(ls.tvef))],
                                      labels=labels)
    }
    if (!missingfill & CI) {
      plt <- plt + scale_fill_manual(NULL, values=fill, labels=labels)
    } else if (CI) {
      plt <- plt + scale_fill_manual(NULL, values=defaultcols[1:length(names(ls.tvef))],
                                     labels=labels)
    }
    if (!missingtitle) plt <- plt + ggtitle(title)
    plt <- plt + guides(linetype=guide_legend(nrow=1), fill=guide_legend(nrow=1), 
                        color=guide_legend(nrow=1)) +
      theme_bw() +
      theme(plot.title=element_text(hjust=0),
            panel.background=element_blank(), panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(), panel.border=element_blank(),
            axis.line=element_line(color="black"),
            axis.title=element_text(size=18, margin=margin(t=0,r=0,b=0,l=0)),
            axis.text=element_text(size=14), text=element_text(size=14),
            legend.title=element_blank(), legend.text=element_text(size=14),
            legend.position=c(0.5, 1), legend.box="horizontal")
    if (save) {
      return(plt)
    } else {
      plt
    }
  }
}

confint.surtiver <- function(fit, times, parm, level=0.95) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  if (missing(times)) {
    times <- fit$times
  } else {
    if (!is.numeric(times) | min(times)<0) stop("Invalid times!")
  }
  if (!is.numeric(level) | level[1]>1 | level[1]<0) stop("Invalid level!")
  level <- level[1]
  times <- times[order(times)]
  times <- unique(times)
  spline <- attr(fit, "spline"); degree <- attr(fit, "degree.spline")
  knots <- fit$internal.knots; nsplines <- attr(fit, "nsplines")
  method <- attr(fit, "control")$method
  term.ti <- names(fit$tief); term.tv <- rownames(fit$ctrl.pts)
  if (missing(parm)) {
    parm <- c(term.tv, term.ti)
  } else if (length(parm)>0) {
    indx <- pmatch(parm, c(term.tv, term.ti), nomatch=0L)
    if (any(indx==0L))
      stop(gettextf("%s not matched!", parm[indx==0L]), domain=NA)
  } else stop("Invalid parm!")
  rownames.info <- c(rep(term.tv, each=nsplines), term.ti)
  if (method=="Newton") {
    invinfo <- solve(fit$info)
  } else if (method=="ProxN") {
    invinfo <- solve(fit$info+diag(sqrt(.Machine$double.eps),dim(fit$info)[1]))
  }
  parm.ti <- intersect(parm, c(term.ti))
  parm.tv <- intersect(parm, c(term.tv))
  quant.upper <- qnorm((1+level)/2)
  ls <- list()
  if (length(parm.ti)!=0) {
    est.ti <- fit$tief[term.ti%in%parm.ti]
    se.ti <- c(sqrt(diag(as.matrix(invinfo[rownames.info%in%parm.ti,
                                           rownames.info%in%parm.ti]))))
    mat.ti <- cbind(est.ti, est.ti-quant.upper*se.ti, est.ti+quant.upper*se.ti)
    colnames(mat.ti) <- 
      c("est", paste0(round(100*c(1-(1+level)/2,(1+level)/2),1),"%"))
    rownames(mat.ti) <- parm.ti
    ls$tief <- mat.ti
  }
  if (length(parm.tv)!=0) {
    if (spline=="B-spline") {
      bases <- splines::bs(times, degree=degree, intercept=T, knots=knots, 
                           Boundary.knots=range(fit$times))
      ctrl.pts <- matrix(fit$ctrl.pts[term.tv%in%parm.tv,], ncol=nsplines)
      ls$tvef <- lapply(parm.tv, function(tv) {
        est.tv <- bases%*%ctrl.pts[parm.tv%in%tv,]
        se.tv <- sqrt(apply(bases, 1, function(r) {
          idx <- rownames.info%in%tv
          return(t(r)%*%invinfo[idx, idx]%*%r)}))
        mat.tv <- cbind(est.tv, est.tv-quant.upper*se.tv, 
                        est.tv+quant.upper*se.tv)
        colnames(mat.tv) <- 
          c("est", paste0(round(100*c(1-(1+level)/2,(1+level)/2),1),"%"))
        rownames(mat.tv) <- times
        return(mat.tv)
      })
      names(ls$tvef) <- parm.tv
    } else if (spline=="P-spline") {
      
    }
  }
  return(ls)
}

AIC.surtiver <- function(fit, k=2) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  if (!is.numeric(k)) stop("Invalid k!")
  return(k*length(c(fit$ctrl.pts, fit$tief))-2*fit$logplkd)
}

BIC.surtiver <- function(fit) {
  return(AIC.surtiver(fit, k=log(length(fit$times))))
}

cox.zph <- function(formula, data, ...) {
  Terms <- terms(formula, specials=c("tv", "strata", "offset"))
  if(attr(Terms, 'response')==0) stop("Formula must have a Surv response!")
  factors <- attr(Terms, 'factors')
  terms <- row.names(factors)
  idx.r <- attr(Terms, 'response')
  idx.o <- attr(Terms, 'specials')$offset
  idx.str <- attr(Terms, 'specials')$strata
  idx.tv <- attr(Terms, 'specials')$tv
  if (is.null(idx.tv)) stop("No variable specified with time-variant effect!")
  term.ti <- terms[setdiff(1:length(terms), c(idx.r, idx.o, idx.str, idx.tv))]
  term.time <- gsub(".*\\(([^,]*),\\s+([^,]*)\\)", "\\1", terms[idx.r])
  term.event <- gsub(".*\\(([^,]*),\\s+([^,]*)\\)", "\\2", terms[idx.r])
  term.o <- if (!is.null(idx.o)) terms[idx.o] else NULL
  term.str <- if(!is.null(idx.str)) terms[idx.str][1] else NULL
  term.tv <- gsub(".*\\((.*)\\)", "\\1", terms[idx.tv])
  print(paste0("Surv(",term.time,",", term.event, ")~",
               paste(c(term.tv, term.ti, term.o, term.str), 
                     collapse="+")))
  `Surv` <- survival::`Surv`; `strata` <- survival::`strata`
  fmla <- as.formula(paste0("Surv(",term.time,",", term.event, ")~",
                            paste(c(term.tv, term.ti, term.o, term.str), 
                                  collapse="+")))
  mat <- survival::cox.zph(survival::coxph(fmla, data), ...)$table
  res <- matrix(mat[rownames(mat)%in%term.tv,], ncol=3)
  rownames(res) <- dimnames(mat)[[1]][rownames(mat)%in%term.tv]
  colnames(res) <- dimnames(mat)[[2]]
  return(res)
}
# baseline plot: smoothed and unsmoothed
basehazard <- function(fit, ...) {
  if (missing(fit)) stop ("Argument fit is required!")
  if (class(fit)!="surtiver") stop("Object fit is not of class 'surtiver'!")
  c(fit$hazard[[1]])
}