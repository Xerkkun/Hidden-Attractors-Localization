import mpmath as mp
mp.mp.dps = 80
alpha = mp.mpf('0.98')
h = mp.mpf('0.005')

vals = {
 "G2mA": mp.gamma(2-alpha),
 "G1pA": mp.gamma(1+alpha),
 "G1p2A": mp.gamma(1+2*alpha),
 "G1p3A": mp.gamma(1+3*alpha),
 "hA": h**alpha,
 "h1mA": h**(1-alpha),
 "inv_hG": 1/(h*mp.gamma(2-alpha)),
 "c2": (1/(2*mp.gamma(1+alpha)))**(1/alpha),
 "c3": (1/(4*mp.gamma(1+alpha)))**(1/alpha),
}
for k,v in vals.items():
    print(k, mp.nstr(v, 25))  # 25 dígitos impresos
