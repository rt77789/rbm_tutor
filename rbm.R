

sigmoid <- function(x, A) {
	#print(paste('dim(x)=', dim(x), sep=''))
	res = t(apply(x, 1, function(r) { 1./(1 + exp(-r * A)) }))
	#print(paste('dim(res)=', dim(res), sep=''))
	res
}

## Generate a matrix of size(row, col), element of which is random generated from normal distribution.
rnorm_matrix <- function(row, col) {
	matrix(rnorm(row*col), nrow=row, ncol=col)
}

rbm_sample <- function(eps = 0.1, epoch = 4000, momentum = 0.7, cost = 0.00001, sig = 0.2) {
#eps = 0.2
#epoch = 4000
#momentum = 0.7
#cost = 0.00001
#sig=0.2
#
	data = read.table('~/code/python/points.data', header=TRUE, sep=',')
	data = as.matrix(data[,1:2])

	## initialize the w,b.
	nnum = nrow(data)
	mnum = ncol(data)
	hnum = 4 

	Av = rep(1, mnum+1) * 0.1
	Ah = rep(1, hnum+1)

	W = matrix(rnorm((mnum+1) * (hnum+1))/10, nrow=mnum + 1, ncol=hnum+1)
	W[1,] = W[,1] = 0

	dW = rnorm_matrix(nrow(W), ncol(W)) / 1000

	V = cbind(rep(1, nnum), data)


	for(i in 1:epoch) {
		## positive
		pos_h_prob = sigmoid(V %*% W + sig*rnorm_matrix(nrow(V), ncol(W)), Ah);
		pos_h_prob[,1] = 1
		pos_h_state = t(apply(pos_h_prob, 2, rbinom, n=nnum, size=1))
		pos_gradient = t(V) %*% pos_h_prob
		apos = t(apply(t(apply(pos_h_prob, 1, function(x) { x * x } )), 2, sum))

		## negative, CD1 is not good enough, so we try CD-K.
		## 1. active from hidden to visible.
		## 2. active from visible to hidden.
		neg_h_prob = pos_h_prob
		for(k in 1:1) {
			neg_v_prob = sigmoid(neg_h_prob %*% t(W) + sig*rnorm_matrix(nrow(neg_h_prob), nrow(W)), Av)
			neg_v_prob[,1] = 1  ## fix the bias coefficient as 1.
			neg_h_prob = sigmoid(neg_v_prob %*% W + sig*rnorm_matrix(nrow(neg_v_prob), ncol(W)), Ah)
		}
		aneg = t(apply(t(apply(neg_h_prob, 1, function(x) { x * x } )), 2, sum))

		neg_gradient = t(neg_v_prob) %*% neg_h_prob

		dW = dW * momentum + eps * ( (pos_gradient - neg_gradient) / nnum - cost * W)
		W = W + dW
		#W = W + eps * ((pos_gradient - neg_gradient) / nnum - cost*W)+ rnorm_matrix(nrow(W), ncol(W))/1000 * momentum

		Ah = Ah + eps * (apos - aneg) / (nnum * Ah * Ah)

		err = sum((V - neg_v_prob)**2) / (mnum * nnum)
		print(paste('epoch: ', as.character(i), ', err=', as.character(err)))
	}

	train.data = data
	data = matrix(runif(2*n, 0.1, 1), nrow=n, ncol=2)

	neg_v_prob = cbind(rep(1, nrow(data)), data)

	## negative, CD1 is not good enough, so we try CD-K.
	## 1. active from hidden to visible.
	## 2. active from visible to hidden.
	for(k in 1:30) {
		neg_h_prob = sigmoid(neg_v_prob %*% W + sig*rnorm_matrix(nrow(neg_v_prob), ncol(W)), Ah)
		neg_h_prob[,1] = 1

		neg_v_prob = sigmoid(neg_h_prob %*% t(W) + sig*rnorm_matrix(nrow(neg_h_prob), nrow(W)), Av)
		neg_v_prob[,1] = 1  ## fix the bias coefficient as 1.
	}
	recon = neg_v_prob

	par(mfrow=c(1,2))
	plot(recon[,2], recon[,3], xlim=c(0,1), ylim=c(0,1))
	plot(train.data[,1], train.data[,2], xlim=c(0,1), ylim=c(0,1))

}
