import numpy as np
import math
import csv
import time

from scipy.sparse import lil_matrix

N = int(input('N: '))
T = int(input('T: '))
arr_rate = float(input('arrival rate (lambda): '))
pb = float(input('pb: '))
ps = 1-pb
lambda_s = (arr_rate*ps)
lambda_b = (arr_rate*pb)
mu_s = float(input('µs: '))
mu_b = float(input('µb: '))

start_sim = time.time()

d1 = round(N + ((N*N)/(2*T)) - (N/2) + (N/T) + 1)
d2 = round((T)*(N/T) + 1)
#print(d1)
#print(d2)

q_dim = d1+(d2*(T+2))
q = lil_matrix((q_dim,q_dim))
#q = csc_matrix(q)
#print(q)

def access_q(n1,s1,b1,n2,s2,b2):
    global q
    if n1 == 0:
        pass
    elif n2 == 0:
        pass
    elif n1 == T+1:
        pass
    else:
        return ((-1,-1))
    
    q_dim = ((q.shape[0]-d1)/d2)

    row = access_map[(s1,b1)]
    col = access_map[(s2,b2)]
    if (n1>0):
        row = d1 + ((n1-1)*d2) + access_map_v2[(s1,b1)]
    if (n2>0):
        col = d1 + ((n2-1)*d2) + access_map_v2[(s2,b2)]
    return((row,col))


access_map = dict()
i = 0
for s in range(N+1):
    for b in range(round(N/T)+1):
        if (s+(b*T)) <= N:
            access_map[(s,b)] = i
            i = i+1

access_map_v2 = dict()
i = 0
for s in range(N+1):
    for b in range(round(N/T)+1):
        if (s+(b*T)) > (N-T) and (s+(b*T)) <= N:
            access_map_v2[(s,b)] = i
            i = i+1

def init_transitions_arr(dim):
    global q
    for n in range(dim):
        for s in range(N+1):
            for b in range (N+1):
                    
                if n > 0 and (s,b) in access_map_v2:
                    r,c = access_q(n,s,b, n+1,s,b)
                    q[r,c] = arr_rate
                elif n == 0:
                    if (s+(b*T)) == N:
                        r,c = access_q(0,s,b, 1,s,b)
                        q[r,c] = arr_rate
                    elif (s+(b*T)) <= (N-T):
                        r,c = access_q(0,s,b, 0,s+1,b)
                        q[r,c] = lambda_s
                        r,c = access_q(0,s,b, 0,s,b+1)
                        q[r,c] = lambda_b
                    elif (s+(b*T)) > (N-T) and (s+(b*T)) < N:
                        r,c = access_q(0,s,b, 0,s+1,b)
                        q[r,c] = lambda_s
                        r,c = access_q(0,s,b, 1,s,b)
                        q[r,c] = lambda_b

def init_transitions_srv(dim):
    global q
    for n in range(dim):
        for s in range(N+1):
            for b in range (N+1):
                F = N - (b*T) - s

                if n > 0 and (s+(b*T)) == N:
                    K = min(n,T)
                    if s > 0:
                        r,c = access_q(n,s,b, n-1,s,b)
                        q[r,c] = q[r,c] + (s*mu_s*ps)

                        r,c = access_q(n,s,b, n,s-1,b)
                        q[r,c] = q[r,c] + (s*mu_s*pb)
                    if b > 0:
                        r,c = access_q(n,s,b, n-1,s,b)
                        q[r,c] = q[r,c] + (b*mu_b*pb)

                        for i in range(1,K):
                            r,c = access_q(n,s,b, n-i,s+i,b-1)
                            q[r,c] = q[r,c] + (b*mu_b*math.pow(ps,i)*pb)

                        r,c = access_q(n,s,b, n-K,s+K,b-1)
                        q[r,c] = q[r,c] + (b*mu_b*math.pow(ps,K))


                elif n > 0 and F > 0 and F < (T-1):
                    K = min(F,n-1)
                    if s > 0:
                        r,c = access_q(n,s,b, n,s-1,b)
                        q[r,c] = q[r,c] + (s*mu_s)
                    if b > 0:
                        for i in range(0,K):
                            r,c = access_q(n,s,b, n-i-1,s+i,b)
                            q[r,c] = q[r,c] + (b*mu_b*math.pow(ps,i)*pb)

                        r,c = access_q(n,s,b, n-K-1,s+K,b)
                        q[r,c] = q[r,c] + (b*mu_b*math.pow(ps,K))

                elif n > 0 and F == (T-1):
                    K = min(F,n-1)
                    if s > 0:
                        r,c = access_q(n,s,b, n-1,s-1,b+1)
                        q[r,c] = q[r,c] + (s*mu_s)
                    if b > 0:
                        for i in range(0,K):
                            r,c = access_q(n,s,b, n-i-1,s+i,b)
                            q[r,c] = q[r,c] + (b*mu_b*math.pow(ps,i)*pb)

                        r,c = access_q(n,s,b, n-K-1,s+K,b)
                        q[r,c] = q[r,c] + (b*mu_b*math.pow(ps,K))

                elif n == 0 and (s+(b*T)) <= N:
                    if s > 0:
                        r,c = access_q(0,s,b, 0,s-1,b)
                        q[r,c] = q[r,c] + (s*mu_s)
                    if b > 0:
                        r,c = access_q(0,s,b, 0,s,b-1)
                        q[r,c] = q[r,c] + (b*mu_b)

def fill_diagonal():
    for i in range(q.shape[0]):
        q[i,i] = -(np.sum(q[i])-q[i,i])
    

init_transitions_arr(T+2)
init_transitions_srv(T+2)
fill_diagonal()

#q = csr_matrix(q).toarray()

diag = d1 + (d2*T)

Bs = []
for i in range(1,T+1):
    block = q[diag:diag+d2, diag-(d2*i):diag-(d2*i)+d2].toarray()
    Bs.append(block)

L = q[diag:diag+d2, diag:diag+d2].toarray()
F = q[diag:diag+d2, diag+d2:diag+d2+d2].toarray()
L_prime = q[0:d1,0:d1].toarray()
F_prime = q[0:d1,d1:d1+d2].toarray()
Bs_prime = [q[d1:d1+d2,0:d1].toarray()]
for i in range(T-1):
    Bs_prime.append(q[d1+((i+1)*d2) : d1+((i+1)*d2)+d2 , 0:d1].toarray())
    
print("Generating time --- %s seconds ---" % (time.time() - start_sim))

L_inv = np.linalg.inv(L)
V = np.matmul(F,L_inv)
Ws = []
for B in Bs:
    W = np.matmul(B,L_inv)
    Ws.append(W)
Ws

Rs = [np.zeros((d2,d2))]
diff = 1
i = 0
while diff > 1e-12:    
    R = -V
    j = 2
    for W in Ws:
        R = np.subtract(R,np.matmul(np.linalg.matrix_power(Rs[i],j),W))
        j += 1
    diff = R[0][0] - Rs[-1][0][0]
    Rs.append(R)
    i += 1
R = Rs[-1]
print(len(Rs))

#
tl = L_prime
tr = F_prime
bl = Bs_prime[0]
for i in range(1,len(Bs_prime)):
    b = Bs_prime[i]
    bl = np.add(bl,np.matmul(np.linalg.matrix_power(R,i),b))
br = L
#i = 1
for i in range(len(Bs)):
    b = Bs[i]
    br = np.add(br,np.matmul(np.linalg.matrix_power(R,i+1),b))
'''for b in Bs:
    br = np.add(br,np.matmul(np.linalg.matrix_power(R,i),b))
    i += 1'''


#
bc = np.append(tl,tr,axis=1)
bcb = np.append(bl,br,axis=1)
bc = np.append(bc,bcb,axis=0)
bc[0][-1] = 1
for r in bc[1:]:
    r[-1] = 0

#
a = np.transpose(bc)
b = np.zeros(d1+d2)
b[-1] = 1
raw_pi = np.linalg.solve(a,b)

#
ir = np.linalg.inv(np.subtract(np.identity(R.shape[0]),R))
r = np.matmul(raw_pi[d1:],ir)
alfa = np.sum(raw_pi[:d1]) + np.sum(r)
pi = raw_pi/alfa
pis = [pi[:d1],pi[d1:]]
for i in range(2,max(N,100)):
    pis.append(np.matmul(pis[i-1],R))
    
np.savetxt("q.csv", q.toarray(), delimiter = ",")
np.savetxt("L.csv", L, delimiter = ",")
np.savetxt("F.csv", F, delimiter = ",")
np.savetxt("L_prime.csv", L_prime, delimiter = ",")
np.savetxt("F_prime.csv", F_prime, delimiter = ",")
i = 1
for b in Bs:
    np.savetxt("B"+str(i)+".csv", b, delimiter = ",")
    i+=1
i = 1
for bp in Bs_prime:
    np.savetxt("B"+str(i)+"_prime.csv", bp, delimiter = ",")
    i+=1

print("Total time --- %s seconds ---" % (time.time() - start_sim))