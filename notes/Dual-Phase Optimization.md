Let $\rho(\mathbf{\theta}) = \ket{\psi(\mathbf{\theta})}\bra{\psi(\mathbf{\theta})}$ be the shared state prepared before the players separate themselves. Additionally, $\phi$ parameterizes the measurement settings $\cal{M(\phi)}$. Let $\cal{M}_{iq}(\phi_{iq})$ denote the gate applied by player $i$ upon receiving question $q$. In general, this operator could act on multiple qubits in the case of games requiring non-dichotomic answers ($a \in \{ 0, 1\}$). The $q$ subscript on $\mathcal{M}$ exists since in principle each player could choose different generators depending on the question (e.g. $R_y$ if $q=0$ and $R_x$ if $q=1$).

There are 2 ways we can optimize the non-local game:

- **Inequality/Bell violation**: $\cal{B} \ge \cal{I}_c$ where $\cal{B}$ is the quantum violation and $\cal{I}_c$ is some classical bound.
- **Game value**: $W$ is the value of the game following the rules $V(a | q)$
$$W = \sum\limits_{q}p(q)\sum\limits_{a}V(a|q)p(a|q),$$
where $q \in Q$ is the set of all possible question combinations, and $a \in A$ is the set of all possible responses.

### Bell Optimization

For a _fixed_ measurement setting $\phi$, we can write the quantum violation operator
$$\cal{B} = \sum\limits_{q} c_{q}~ M(q; \phi),$$
where
$$\cal{M}(q; \phi) = M_1(\phi_{q_1}^1) \otimes\cdots\otimes M_n(\phi_{q_n}^n)$$
since each player chooses their measurement angle $\phi_{q_i}^i$ independently based on the question $q_i$ they receive. The largest eigenvalue of $\cal{B}$ is the maximal quantum violation. Conversely, we can define $H = -\cal{B}$ to get a time-independent hamiltonian whose ground state corresponds to the maximal violation. This is solvable using VQE routines (e.g., ADAPT).

**(Needs work)** The above formula is probably incorrect, and I cannot find a recipe for constructing arbitrary hamiltonians from games in papers. Everyone seems to use 3 different NLGs and call it a day, borrowing the inequalities.

### Value Optimization

Let the value of the game be
$$
\begin{align}
W &= \sum\limits_{q}p(q) \sum\limits_a V(a|q)\pi(a|q)\\
&= \sum\limits_{q,a}w_{qa}~\pi(a|q),
\end{align}$$
where $\pi(a|q) = p(a|q)$ is the policy of the players in the game and $w_{qa} = p(q)V(a|q)$. One such quantum strategy is
$$
\begin{align}
\pi(\mathbf{a|q}) &= \mathrm{tr}\left[\rho(\theta) \mathcal{M}_{a|q} \right]\\
&= \mathrm{tr}\left[\rho(\theta) \bigotimes_{i}\mathcal{M}_{a|i,q}(\phi_{iq}) \right],
\end{align}
$$
where $\cal{M}_{a|i,q}(\phi_{iq})$ denotes a parameterized projective measurement for answer $a\in A$ applied by player $i$ upon receiving question $q \in Q$ from the referee.

**(Warning)** Note that this is slight abuse of notation since $\mathcal{M_{a|q}}$ denotes the composite measurement operator produced (independently) by all players for answer $\mathbf{a} = [a_{1}~a_{2}\cdots]$ upon receiving question $\mathbf{q} = [q_{1}~q_2\cdots]$. In contrast, the individual measurement operators $\mathcal{M}_{a|i,q}$ is the operator player $i$ uses to give the probability they respond with answer $a=\mathbf{a}_{i}$ for question $q=\mathbf{q}_i$. 

This measurement operator can be expressed as
$$\mathcal{M}_{a|q} = U^\dagger_q(\phi_{q})P_{a}U_q(\phi_{q}),$$
where $P_{a} = \ket{a}\bra{a}$ is the projector onto joint answer $\mathbf{a}$, $U_q(\phi_q)$ is the unitary gate
$$U_{q}(\phi_{q}) = U_{1q_{1}}(\phi_{1q_{1}})~\otimes~\cdots~\otimes U_{nq_{n}}(\phi_{nq_{n}}),$$
and $\phi_{q} = [\phi_{1q_{1}}~\phi_{2q_{2}}~\cdots\phi_{nq_{n}}]$ is the parameter vector consisting of the chosen measurement angles by each player for question $\mathbf{q}$. This reduces the expression above to
$$\pi_{\theta\phi}(\mathbf{a}|\mathbf{q}) = |\braket{\psi(\theta, \phi_{\mathbf{q}})|\mathbf{a}}|^{2},$$
for a pure state $\rho = \ket{\psi}\bra{\psi}$. Substituting the policy into the value of the game,
$$
\begin{align}
W &= \sum\limits_{qa}w_{qa} \pi(a|q) \\
&= \sum\limits_{qa} w_{qa}~ \mathrm{tr}\left[\rho \mathcal{M}_{a|q} \right] \\
&= \mathrm{tr}\left[\rho \sum\limits_{qa}w_{qa}\mathcal{M}_{a|q} \right] \\
&= \mathrm{tr}\left[\rho(\theta) V(\phi) \right],
\end{align}
$$
where $\mathcal{M}_{a|q} = \bigotimes_{i} U_{iq}^{\dagger}(\phi_{iq}) P_{a_{i}} U_{iq}(\phi_{iq})$.

**(Needs work/unsure)** This suggests the existence of a value operator $V(\mathbf{\phi})$ that gives the value of a game, and therefore the maximal eigenvalue should be the largest win rate. Taking $\mathcal{H} = -V$ should give a recipe for constructing hamiltonians from arbitrary non-local games. Empirically, I tried constructing such a hamiltonian for CHSH game, but the eigenvalues were the classical strategies: either 0.25 or 0.75.

```ad-info
title: Example: CHSH Game

Alice has 4 possible operators, $\ket{a}\bra{a}R_y(\alpha_x)$ for $x, a \in \{0, 1\}$. Similarly for Bob, $\ket{b}\bra{b}R_y(\beta_y)$.

When it's time to actually execute this on a quantum computer, however, we don't take the expectation value of all 16 possible combinations. Instead, the projections $Z_{a}, Z_{b}$ happen naturally by measuring in the computational basis. We would apply

$$\rm{tr}\left[ \rho \left(R_y(\alpha_{x}) \otimes R_y(\beta_{y})\right) \right]$$

and collect statistics on our outcomes $\{\ket{00}, \ket{01}, \cdots \}$
```

The matrix $\phi = [\phi_{iq}]$ has $N \times q$ parameters (where $q$ is the number of possible questions), and we can optimize these along with $\theta$ to maximize our odds of winning the game using the gradient $\nabla_{\theta\phi}W(\theta,\phi)$.

To optimize $\theta$,
$$
\begin{align}
\nabla_{\theta}W &= \mathrm{tr}\left[\nabla_{\theta}\rho(\theta)V(\phi) \right] \\
&= \sum\limits_{qa}w_{qa}~\mathrm{tr} \left[\nabla_{\theta}\rho(\theta) \cal{M}_{a|q}\right],
\end{align}
$$
which of course depends on our particular ansatz. This gradient is computable with parameter-shift rules (PSR), $\frac{\partial W}{\partial \theta_{i}}= c\left[ W(\theta + s\hat{e}_{i}) - W(\theta - s\hat{e}_{i})\right]$. For fixed $\phi$, this seems identical to VQE except for the weighting coefficients.

To optimize $\phi$,
$$
\begin{align}
\frac{\partial W}{\partial \phi_{iq}} &= \sum\limits_{qa}w_{qa}~\mathrm{tr}\left[\rho \frac{\partial \cal{M}}{\partial \phi_{iq}} \right]\\
&= \sum\limits_{qa} w_{qa}~\mathrm{tr} \left[\rho(\theta, \phi_{q})~i\left[P_{a}, g_{iq} \right]\right],
\end{align}$$where $\rho(\theta, \phi_{q}) = U^{\dagger}_{q}(\phi_{q})\rho(\theta)U_{q}(\phi_{q})$ is the combined shared state and measurement rotation layers, and $g_{iq}$ is the generator for the unitary gate $U_{iq} = e^{i\phi_{iq}g_{iq}}$ of player $i$ that only acts on the Hilbert subspace $\cal{H}_i$. This gradient expression is real - to see this, note that $i[P_{a}, g_{iq}]$ defines a hermitian observable because $P_a$ and $g_{iq}$ are hermitian as well. This derivative should also be computable using PSR on an actual quantum device.

As a mathematical note,
$$
\begin{align}
U_q(\phi_{q}) &= \bigotimes_{i} U_{iq}(\phi_{iq}) \\
&= \bigotimes_{i} e^{i\phi_{iq}g_{iq}} \\
&= e^{i\sum\limits_{i}\phi_{iq}g_{iq}}\\
&= e^{iG_{q}}
\end{align},
$$
where $G_q$ is the generator for the full $N$-player unitary measurement layer conditioned on question $\mathbf{q}$.

**Are these gradients well-defined/correct?**

Let $f(a, q; \theta, \phi_{q}) = |\langle\psi(\theta, \phi_{q})|a\rangle|^{2} \in \mathbb{R}$. Then,
$$W = \sum\limits_{qa}w_{qa}f(a,q;\theta, \phi_{q})$$
From this it's very easy to see that the gradients must be real, since $f(\cdot~| \theta + \delta \theta, \phi_{q})$ and $f(\cdot~| \theta, \phi_{q} + \delta \phi)$ are also real.

### Dual-Phase Optimization

In general, we could just do gradient descent on all the parameters $\left[\theta~\phi \right]$. However, because fixing the measurement settings $\phi$ gives us a VQE problem, this motivates an iterative procedure that has 2 phases:

1. Fix $\phi$. We prepare the optimal state $\rho$ **for those measurement settings** using a VQE algorithm (e.g., ADAPT for small ansatze).
2. Tune $\phi$ for the state we prepared in step 1.

Algorithm:
	Initialize $\phi^{(0)}$ randomly
	while $\Delta W > \epsilon$ do
		$\rho(\theta^{(k+1)}) = \mathrm{VQE}(H(\phi^{(k)}))$
		$\mathrm{optimize}(\phi^{(k+1)} | \rho(\theta^{(k+1)}))$

**(Note)** We could also skip the VQE part and just optimize $W$ for fixed ansatz, or modify ADAPT to construct ansatz for $W$ if creating hamiltonians is too hard.

Phase 1 is well-defined and behaves fine except for barren plateaus and other VQE problems.

Phase 2 could use some proof.
- If we are near an optimal set of parameters, does optimizing $\phi$ take us towards the maximum?
- After optimizing $\phi$, is $\rho$ still a ground state?
	- Almost surely not. There's no reason that $H(\phi^{(k+1)}) = H(\phi^{(k)} + \delta \phi)$ should share ground states with $H(\phi^{(k)})$.
- How smooth is the landscape of $\phi$?
	- The derivative is continuous.
- After optimizing $\phi$, are the minima $\phi^*$ the same?