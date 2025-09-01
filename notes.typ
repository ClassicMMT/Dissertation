= Notes

== Goals
- Develop an adversarial framework (with metric)
- For testing how good a model is, specifically out-of-distribution
- Find a boundary to test the applicability domain

== What I need to do
- Define the applicability domain - defined as an indicator for when a predictor acts inside the domain it works reliably on. (applicability, reliability, decidability)
- Figure out how to find the applicability domain


== Stuff to research
+ when test distribution is different from train
+ look into bias
+ overfitting
+ adversarial regions

== New ideas
+ Explore the behaviour of the wasserstein distance. Cases with "fake" attacks, where say only the y coords move but not the x. Cases where both need to move but one shrinks (concave decision boundary). Cases where it expands (convex decision boundary).
+ Try different models
+ what happens if the data is not linearly separable? what if it is?
+ try with more features (no plots) and train/test sets.
+ what happens to the wasserstein distance when the variance is changed?
+ do these points continue to hold when we increase the dimensionality?
+ look into optimal transport
+ Impact of class imbalance on wasserstein distance

== Older ideas
+ You can try to approximate a cdf and see if it works in the distance metrics.
+ You can also try create your own metric.


== Distance metrics

- Total variation distance (tv distance)
  - Measures maximum difference between two disributions
  - Symmetric
  - Between 0 and 1
- Hellinger Distance
  - Symmetric
  - Between 0 and 1
  - Good for non-normal distributions
- Cosine similarity?
  - can measure distributions in terms of direction
- Kolmogorov-Smirnov statistic
  - non-parametric test that measures the distance between the empirical cumulative distibution functions
- Mahalanobis distance
  - only to compare single point to a distribution
- Cramer-von Mises criterion
  - Symmetric and based on CDFs

== Distance metrics tried
- Wasserstein Distance (earth mover distance)
  - non-negative and symmetric
- Kullback-Leibler Divergence
  - Not symmetric
  - Can be infinite
  - Says its only for exponential families
- Jensen-Shannon divergence (JS divergence)
  - Symmetric version of KL divergence
- Bhattacharyya distance
  - Symmetric
  - 0 to infinity, where 0 is identical

== Questions for Joerg

- Can i compare a sample against a fake cdf (cumsum over another sample)? Basically this would be treating one as a distribution. Asking about the KS test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html


== Other stuff

https://github.com/KatDost/Imitate
https://dx.doi.org/10.1109/ICDM50108.2020.00115
https://ml.auckland.ac.nz/summer-project-auditing-machine-learning-models-quantifying-reliability-using-adversarial-regions/

Ambra Demontis presentation on adversarial ML:
* https://www.youtube.com/watch?v=DKWs9lOyiLg
