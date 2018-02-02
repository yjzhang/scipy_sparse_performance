import time

import numpy as np
from scipy import sparse

# mprof

n_trials = 100

m = 20000
n = 20000

# coo to csc

coo_0 = sparse.random(m, n, format='coo')

t0 = time.time()
for i in range(n_trials):
    csc_0 = sparse.csc_matrix(coo_0)
print('time for 100 coo-to-csc conversions: {0}'.format(time.time() - t0))

# csc to csr

t0 = time.time()
for i in range(n_trials):
    csr_0 = sparse.csr_matrix(csc_0)
print('time for 100 csc-to-csr conversions: {0}'.format(time.time() - t0))

# csc to coo

t0 = time.time()
for i in range(n_trials):
    coo_1 = sparse.coo_matrix(csc_0)
print('time for 100 csc-to-coo conversions: {0}'.format(time.time() - t0))


# slicing

# csc slice by column

t0 = time.time()
for i in range(n_trials):
    csc_sub0 = csc_0[:,0]
print('time for 100 csc column slices: {0}'.format(time.time() - t0))

# csc slice by row

t0 = time.time()
for i in range(n_trials):
    csc_sub0 = csc_0[0,:]
print('time for 100 csc row slices: {0}'.format(time.time() - t0))

# csr slice by column

t0 = time.time()
for i in range(n_trials):
    csr_sub0 = csr_0[:,0]
print('time for 100 csr column slices: {0}'.format(time.time() - t0))

# csr slice by row

t0 = time.time()
for i in range(n_trials):
    csr_sub0 = csr_0[0,:]
print('time for 100 csr row slices: {0}'.format(time.time() - t0))

# means

# mean(0) is mean along each column
t0 = time.time()
for i in range(n_trials):
    csr_sub0 = csr_0.mean(0)
print('time for 100 csr col means: {0}'.format(time.time() - t0))

t0 = time.time()
for i in range(n_trials):
    csr_sub0 = csr_0.mean(1)
print('time for 100 csr row means: {0}'.format(time.time() - t0))

t0 = time.time()
for i in range(n_trials):
    csc_sub0 = csc_0.mean(0)
print('time for 100 csc col means: {0}'.format(time.time() - t0))

t0 = time.time()
for i in range(n_trials):
    csc_sub0 = csc_0.mean(1)
print('time for 100 csc row means: {0}'.format(time.time() - t0))


t0 = time.time()
for i in range(n_trials):
    coo_sub0 = coo_0.mean(0)
print('time for 100 coo col means: {0}'.format(time.time() - t0))


# sparse matrix multiplication

n_trials_short = 20

t0 = time.time()

# this is very costly - a big part of it is probably memory allocation,
# since the product is likely a lot denser.
# time for 20 csc*csc products: 36.6816399097
#for i in range(n_trials_short):
#    csc_prod = csc_0.dot(csc_0)
#print('time for 20 csc*csc products: {0}'.format(time.time() - t0))


#t0 = time.time()
#for i in range(n_trials_short):
#    csr_prod = csr_0.dot(csr_0)
#print('time for 20 csr*csr products: {0}'.format(time.time() - t0))


# performance for type conversion

t0 = time.time()
for i in range(n_trials):
    csr_2 = csr_0.astype(int)
print('time for 100 csr type conversions: {0}'.format(time.time() - t0))

# should be identical to above
t0 = time.time()
for i in range(n_trials):
    csc_2 = csc_0.astype(int)
print('time for 100 csc type conversions: {0}'.format(time.time() - t0))



# sparse-to-dense conversions for slice
t0 = time.time()
for i in range(n_trials):
    csc_sub0 = csc_0[:,0]
    dense_0 = csc_sub0.toarray()
print('time for 100 csc_0[:,0].toarray(): {0}'.format(time.time() - t0))
