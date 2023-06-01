runfor:
	mpic++ ./more_use.cu
	mpiexec -n 4 ./a.out res size 13
