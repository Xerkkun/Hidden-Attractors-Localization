import sys
# import numfracpy as nfr
import mittag_leffler as ml
from scipy.special import gamma
import numpy as np

# 3-stage EFORK method for Example 1.

n = len( sys.argv )
if n != 2 :
	print( "Args: N^m_value" )
	sys.exit(1)

N1 = int( sys.argv[1] )
# m = 4
T = 1.0
alpha = 1.0/2 
h=T/N1

w1 = ( 8.0*gamma(1+alpha)**3 * gamma(1+2*alpha)**2 - 6.0*gamma(1+alpha)**3 * \
	gamma(1+3*alpha) + gamma(1+2*alpha)*gamma(1+3*alpha) ) / \
	( gamma(1+alpha)*gamma(1+2*alpha)*gamma(1+3*alpha) )

w2 = 2.0*gamma(1+alpha)**2 *( 4.0*gamma(1+2*alpha)**2 - gamma(1+3*alpha) ) / \
	( gamma(1+2*alpha)*gamma(1+3*alpha) )

w3 = -8.0*gamma(1+alpha)**2 *( 2.0*gamma(1+2*alpha)**2 - gamma(1+3*alpha) )/ \
	( gamma(1+2*alpha)*gamma(1+3*alpha) )

a11 = 1.0/(2*gamma(alpha+1)**2)
a21 = ( gamma(alpha+1)**2 * gamma(2*alpha+1) + 2*gamma(2*alpha+1)**2 - \
	gamma(3*alpha+1) )/( 4*gamma(alpha+1)**2 *
	( 2*gamma(2*alpha+1)**2 - gamma(3*alpha+1) ) )

a22 = - gamma(2*alpha+1)/( 4*( 2*gamma(2*alpha+1)**2 - gamma(3*alpha+1) ) ) 

c2 = ( 1.0/( 2*gamma(1+alpha) ))**(1.0/alpha)
c3 = ( 1.0/( 4*gamma(1+alpha) ))**(1.0/alpha)

# Todas las constantes tienes valores OK

def f0( t, y ) :
	return -y + (t**(4.0-alpha))/gamma(5.0-alpha)

vtn = np.zeros( N1+1 ) 
vyn = np.zeros( N1+1 ) 
vfn1 = np.zeros( N1+1 ) 

n=0
ha = h**alpha
tn = 0.0
yn = 0.0
vtn[n] = tn
vyn[n] = yn
K1 = ha * f0( tn, yn )
K2 = ha * f0( tn + c2*h, yn + a11*K1 )
K3 = ha * f0( tn + c3*h, yn + a22*K2 + a21*K1 )
yn1 = yn + w1*K1 + w2*K2 + w3*K3
tn1 = (n+1)*h
vtn[n+1] = tn1
vyn[n+1] = yn1
f = (tn1**4) * ml.ml( -tn1**alpha, alpha, 5 )
error = abs(yn1 - f)

print( n+1, "error=", error )

# Cálculos para n > 0

def fn( k, t, y ) :
	v = 0.0
	i = 0
	while i < k :
		t0 = vtn[i]
		t1 = vtn[i+1]
		v1 = (t-t0)**(1.0-alpha)
		v2 = (t-t1)**(1.0-alpha)
		a = (vyn[i+1] - vyn[i])/(h*gamma(2-alpha))
		v += a*(v1-v2)
		i += 1

	# return f0(t, y) - v/((1-alpha)*gamma(1-alpha))  
	v3 = f0( t, y ) - v

	return v3 

n = 1
while n < N1 :
	tn = n*h
	
	K1 = ha * fn( n, tn, yn )
	K2 = ha * fn( n, tn + c2*h, yn + a11*K1 )
	K3 = ha * fn( n, tn + c3*h, yn + a22*K2 + a21*K1 )

	yn1 = yn + w1*K1 + w2*K2 + w3*K3

	tn1 = (n+1)*h
	vtn[n+1] = tn1 
	vyn[n+1] = yn1

	f = (tn1**4) * ml.ml( -tn1**alpha, alpha, 5 )
	error = abs( yn1 - f )
	print( n+1, "error=", error )

	yn = yn1
	n += 1
