get_std_coeff <- function(X)
{
	std_ <- apply(X, 2, sd)
	std_[std_ < 1e-16] <- 1.0
	return ( list(apply(X, 2, mean), std_) )
}

standardize <- function(X, std_coeff)
{
	X <- scale(X, std_coeff[[1]], std_coeff[[2]])
	return (X)
}
