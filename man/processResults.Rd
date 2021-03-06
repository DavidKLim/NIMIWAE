% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/processResults.R
\name{processResults}
\alias{processResults}
\title{Process results: return imputation metrics}
\usage{
processResults(
  data = NULL,
  Missing = NULL,
  g = NULL,
  res = NULL,
  data.file.name = "",
  res.file.name = "",
  method = c("MIWAE", "NIMIWAE", "HIVAE", "VAEAC", "MEAN", "MF")
)
}
\arguments{
\item{data}{Data matrix (N x P)}

\item{Missing}{Missingness mask matrix (N x P)}

\item{g}{Training-validation-test split partitioning}

\item{res}{Results object output from NIMIWAE function}

\item{data.file.name}{Path to data file, which contains "data", "Missing", and "g". These inputs need not be specified if data.file.name is specified}

\item{res.file.name}{Path to res file, which contains "res_<method>" results object from the method that was run. "res" need not be specified if res.file.name is specified}

\item{method}{Method used for imputation. "NIMIWAE" is used for this package, but results from competing methods ("MIWAE", "HIVAE", "VAEAC", "MEAN", "MF") can also be processed. See the NIMIWAE_Paper repo for more details}
}
\value{
list of objects: res (original res input), results (named vector of imputation metrics), and call (user-input call to function)
}
\description{
Process results: return imputation metrics
}
\examples{
processResults(data.file.name="Results/CONCRETE/data_MCAR_25.RData", res.file.name="Results/CONCRETE/res_NIMIWAE_MCAR_25_IWAE_rzF.RData", method="NIMIWAE")

}
\references{
\url{https://github.com/DavidKLim/NIMIWAE}
}
\author{
David K. Lim, \email{deelim@live.unc.edu}
}
