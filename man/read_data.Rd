% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prepareData.R
\name{read_data}
\alias{read_data}
\title{Read UCI or Physionet 2012 Challenge Data}
\usage{
read_data(
  dataset = c("Physionet_mean", "Physionet_all", "HEPMASS", "POWER", "GAS", "IRIS",
    "RED", "WHITE", "YEAST", "BREAST", "CONCRETE", "BANKNOTE"),
  ratio = c(6, 2, 2),
  g_seed = 333
)
}
\arguments{
\item{dataset}{String for name of dataset. Valid datasets: "BANKNOTE","....".}

\item{ratio}{Train-valid-test ratio for splitting of observations}

\item{g_seed}{Seed for train-valid-test dataset splitting of observations}
}
\value{
list of objects: data (N x P matrix), classes (subgroups of observations), params (those used for simulating data), and g (partitioning of data into train-valid-test sets)
}
\description{
Read UCI or Physionet 2012 Challenge Data
}
\examples{
read_data(dataset = "BANKNOTE")

}
\references{
\url{https://github.com/DavidKLim/NIMIWAE}
}
\author{
David K. Lim, \email{deelim@live.unc.edu}
}
