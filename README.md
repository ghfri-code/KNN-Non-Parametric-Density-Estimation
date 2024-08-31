# Non Parametric Density Estimation
For this project, we generate a dataset for three classes each with 500 samples from three Gaussian distribution described below:

$$ class1:\quad\mu = \binom{2}{5} \qquad 
\sum =
\begin{pmatrix}
2 & 0 
\\
0 & 2
\end{pmatrix}
$$

$$ class2:\quad\mu = \binom{8}{1} \qquad 
\sum =
\begin{pmatrix}
3 & 1
\\
1 & 3
\end{pmatrix}
$$

$$ class3:\quad\mu = \binom{5}{3} \qquad 
\sum =
\begin{pmatrix}
2 & 1
\\
1 & 2
\end{pmatrix}
$$

Use generated data and estimate the density without pre-assuming a model for the distribution which is done by a non-parametric estimation.
Implement the KNN PDF estimation methods using h=0.09,0.3,0.6. Estimate P(X) and Plot the true and estimated PDF.
### True Density 3D
![true density 3d](https://github.com/Ghafarian-code/KNN-Non-Parametric-Density-Estimation/blob/master/images/Figure_2.png)
### KNN Density 3D
![KNN density 3d](https://github.com/Ghafarian-code/KNN-Non-Parametric-Density-Estimation/blob/master/images/Figure_4.png)

At k=1, the graph is sensitive to noise and it causes discontinuity.
At k = 9, for each x, we considered 9 of its neighbors, and our volume has become larger and the peaks are more specific.
At k = 99, multiclass data and paeks are clearer and the graph is smoother.                                    
