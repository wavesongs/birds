# ⚙️ Physical Model

:::{image} ../../assets/img/model.png
:alt: Physical model
:width: 70%
:align: center
:::


## Syrinx

```{math}
:label: syrinx_general
\frac{dx^2}{dt^2} = \gamma^2[-\alpha (t)-\beta (t) x  + x^2 - x^3] - \gamma (1+ x )x\frac{dx}{dy}
```

```{math}
:label: syrinx_2eq
\begin{align}\label{syrinx_edos}
    & \frac{dx}{dt} = y\nonumber\\
    & \frac{dy}{dt} = \gamma^2[-\alpha (t)-\beta (t) x  + x^2 - x^3] - \gamma (1+ x )xy 
\end{align}
```




## Trache

```{math}
:label: trachea
\begin{gather}\label{trache_edos}
p_i (t) = A y(t) + p_{back}\left( t - \frac{L}{c} \right) \\
p_{back} (t) = -r p_i\left( t - \frac{L}{c} \right)\\
p_{out} (t) = (1-r)p_i\left( t - \frac{L}{c} \right)\\
\frac{d p_{out}}{dt} (t) = \frac{p_{out}(t)- p_{out}(t-dt)}{dt}
\end{gather}
```

## OEC

```{math}
:label: OEC
\begin{align}\label{OEC_edos}
   \frac{d}{dt} p_g &= p_g'= dp_g \nonumber\\
   \frac{d}{dt} dp_g  &= dp_g'= -\frac{1}{C_{OEC} M_g}p_g  - R_{OEC} \left( \frac{1}{M_b} + \frac{1}{M_g}\right) dp_g  \nonumber\\
   &  \qquad \qquad + \frac{1}{M_g}\left( \frac{1}{C_{OEC} } + \frac{R_{OEC} R_b}{ M_b}\right) p_b + \frac{1}{M_g}\frac{dp_{out} }{dt} + \frac{R_{OEC}}{M_g M_b  }  p_{out} \nonumber\\
   \frac{d}{dt} p_b &= p_b' = -  \frac{M_g}{M_b}  p_g' - \frac{R_b}{M_b} p_b + \frac{1}{M_b} p_{out}
   %\\  & \color{myblue} \frac{d\vec{p}}{dt} = \vec{h}(\vec{p},  p_{out}, p_{out}'), \quad \vec{p} = (p_g, p_g', p_b)
\end{align}
```

## Minimization Problem

```{math}
:label: general_minimization_problem
\begin{equation}\label{opt_general}
\begin{aligned}
\underset{ \gamma \in \mathbb{R},\; \alpha,\beta\in \mathbb{R}^n}{\text{min}} &\qquad  ||\hat{SCI}_{real} - \hat{SCI}_{synt} ( \gamma,\alpha,\beta)||_2  + || (\hat{FF}_{real} - \hat{FF}_{synt}(\gamma,\alpha,\beta)||_2 \\
    & \qquad \qquad  - corr(FC_{real},FC_{synt}(\gamma, \alpha, \beta)) \\
    \text { subject to }  & \qquad \gamma \in \Omega_\gamma, \quad  \beta \in \Omega_\beta ,  \quad  \alpha \in \Omega_\alpha
\end{aligned}
\end{equation}
```

## Numerial Solution

:::{image} ../../assets/img/methodology.png
:alt: Physical model
:width: 70%
:align: center
:::


{cite:p}`muscles_role`

## References


```{eval-rst}
.. bibliography:: ../references/articles.bib
   :all:
```