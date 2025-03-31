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

== Distannce metrics tried
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
