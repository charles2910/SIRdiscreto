
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# População Total, N.
N = 1000
# Número inicial de indivíduos infectados e recuperados, I0 e R0.
I0, R0 = 1, 0
# O resto, S0, são os indivíduos ainda suscetpiveis à infecção.
S0 = N - I0 - R0
# Taxa de contato efetivo, alfa, e taxa média de recuperação, gamma.
alfa, gamma = 0.107, 1./14 
# Valores de tempo para avaliar os compartimentos em dias.
t = np.linspace(0, 320, 320)
# O modelo SIR de equações diferenciais.
def deriv(y, t, N, alfa, gamma):
    S, I, R = y
    dSdt = -alfa * S * I / N
    dIdt = alfa * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt
# Vetor de condições iniciais.
y0 = S0, I0, R0
# Integração das equações do modelo SIR no tempo t.
ret = odeint(deriv, y0, t, args=(N, alfa, gamma))
S, I, R = ret.T
# Modelo em tempo discreto do SIR
Sd, Id, Rd = np.zeros(320), np.zeros(320), np.zeros(320)
Sd[0], Id[0], Rd[0] = S0, I0, R0

for i in range(1,320):
    Sd[i] = Sd[i - 1] - (alfa * Sd[i - 1] * Id[i - 1] / N)
    Id[i] = Id[i - 1] - gamma * Id[i - 1] + (alfa * Sd[i - 1] * Id[i - 1] / N)
    Rd[i] = Rd[i - 1] + (gamma * Id[i - 1])

# Plot dos valores contínuos e discretos de S(t), I(t) e R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Suscetíveis')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infectados')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recuperados')
ax.plot(t, Sd/1000, 'c--', alpha=0.5, lw=2, label='[Discreto] Suscetíveis')
ax.plot(t, Id/1000, 'm--', alpha=0.5, lw=2, label='[Discreto] Infectados')
ax.plot(t, Rd/1000, 'y--', alpha=0.5, lw=2, label='[Discreto] Recuperados')
ax.set_title('Modelo SIR')
ax.set_xlabel('Dias')
ax.set_ylabel('População Relativa')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
