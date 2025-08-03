# Entropy-Based Distributions

### Brief Introduction
This repository is an expiremental rpoject related to the Maximum Entropy Principle and how it can be applied. The paper is composed of two parts [Part 1](http://bayes.wustl.edu/etj/articles/theory.1.pdf) and 
[Part 2](http://bayes.wustl.edu/etj/articles/theory.2.pdf). To provide a brief summary: Entropy is a concept that has many different applications depending on the domain, the paper discusses that the there is a link
between the domain of physics and information theory. In physics entropy is used in the 2nd law of thermodynamics which states that the entropy in a closed system will always increase. In information theory the entropy proposed by Shannon is commonly related to the 'uncertainty' or 'randomness' of probabilities. For example given a coin toss outcomes scenario 1 = [H,T] = [0.7,0.3] versus scenario 2 = [H,T] = [0.5,0.5]. Scenario 2 will be more
random as scenario 1 is skewness towards heads, thus more predicable. 
Jaynes in the paper demonstrates that indeed the definition of entropy in relation to thermodynamics and information thoery is the same, such to say that if thermodynamics states that entropy will always increase then it must be true in information theory aswell. Jaynes then proposes a proof to demonstrate how probability distribution functions can be derived by maximizing the entropy and applying constraints.
Here is an example of deriving probability distribution functions using Maximum Entropy Principle [MEP](https://michael-franke.github.io/intro-data-analysis/the-maximum-entropy-principle.html)

### The Problem
There are two main issues with Maximum Entropy Principle:
1. Knowing the constraints of the system implies knowledge of the distribution. 
Suppose we have data that represents a uniform distribution, if we unkowingly constrain the mean, then maximum entropy principle will derive a gaussian distribution. Thus knowing the constraints of the system
are crucial to deriving a correct probability distribution funciton. 
2. Maximum entropy principle creates generalized distributions, rather than unique distributions depending on the data. In other words there must be a method to factor in the data that the distribution is derived from. 

### The Solution
To solve the limitations of maximum entropy principle we create a generic distribution
$
p(x) = \frac{1}{Z} e^{\lambda_1f_1(x)+\lambda_2f_2(x)+...+\lambda_nf_n(x)}
$
1/Z represents the normalization constraint, thus the problem now becomes how can these lambdas which are constraints be solved for, while incorporating the data. 
Our method is two fold.
1. Using Log-likelihood function from MLE. 
This idea is the main approach and the idea is that there is a log-likelihood function which estimatest the parameters of a distribution by taking the log and taking the derivate with respect to that parameter. In our scenario we use numerical approaches to create this derivation. 
The log-likelihood is defined as,
$
 log(\sum_{i=1}^{N} P(x_i|\lambda_i) )
$.
Maximizing the log-likelihood will yield the constraints of the system and thus the distribution of the data with respect to the
2. Fidelity
The other approach which yields similar results is to minimize fidelity which is the measure of the distance between two distributions. If the distributions are equal it will be 1.
Fidelity is defined as,
$
\sum_{i=1}^{N} \sqrt{p_i\times q_i}
$
Where p and q are the probabilities related to the distribution.

### This REPO/Folder Structure 
The primary code explained above is in the folder: Maximum Entropy Principle, in the file combined.py
The visualizer.py is a strictly visualization purposes application to show how changing the lambdas will affect the distribution
/Basis/ was another approach to achieve similar results, but it does not scale well, because it uses least squares on a massive matrix
/data/ is the data containts arbitary datasets this method was used to test, 

### Future Scope
The future and possible expirements could be considered.
1. In our use case I used 4 lagrangian constraints, this is because in most practices I saw that most distributions were derived in four or less constraints. Though it could be interesting to see if more constraints
yields more accurate results.
2. The optimizer used is scipy's optimizer which is effective though it could also be possible to tune lambdas using the effective solver in QiML. Assuming rather than optimizing hyper-parameters, optimize the lambdas.
3. Application in low-level hardware. This project was primarily a proof of concept of how it can be used, though it could be beneficial to create a form of online-learning on low-level baseband instruments. But this involves rewriting the implementation in low-level software such as C/C++.