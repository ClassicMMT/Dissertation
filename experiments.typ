= Pages for Experimental Results

== From Joerg

+ shape of the distribution of the distances
+ adversarial regions
+ detecting overfitting via adversarial examples
+ wild patterns paper from joerg
+ learn what the attacks actually do
+ check the bias paper and other stuff

== Wasserstein Distance

=== Consistency across many datasets and models

TODO

=== Same scale



Suppose there are two distributions where $X_1 ~ N(0, 5)$ and $X_2 ~ 0.5 * N(0,1) + 0.5 * N(5, 1)$; i.e. the scale is the same for both classes along $X_1$; $X_2$ is bimodal. The initialisation is:


```
np.random.seed(123)

x1 = np.random.normal(loc=0, scale=5, size=(200, 1))
x2 = np.vstack((
    np.random.normal(loc=5, scale=1, size=(100, 1)),
    np.random.normal(loc=0, scale=1, size=(100, 1)),
))
x = np.hstack((x1, x2))
y = np.hstack((np.zeros(100), np.ones(100))).astype(int)

normal = SVC(kernel="rbf", random_state=101)
overfit = SVC(kernel="rbf", random_state=102, gamma=50)
```

If we run the `HopSkipJump` attack we obtain the following plots:

#figure(
  image("images/same_scale_fig01.png"),
  caption: [Same scale experiment],
) <fig01>

Where it appears that:
+ for the normal model, the points in both classes need to move quite a lot relative to the blue class in the overfit model.
+ We also see that it may be harder for the `HopSkipJump` attack to actually find the decision boundary for the overfit model.
+ Computing the sum of squares and taking the average across points, actually shows that the points had to move more overall for the normal model relative to the overfit model.
  - We would expect the SS to get larger the more points move (larger distance between distributions), so the normal model should have larger SS and distance. However, this will change if the metrics are calculated by class.
  - When split by classes, the distance the points need to travel, to become adversarial, gets significantly bigger


=== Different Scale

#figure(
  image("images/different_scale_fig01.png"),
  caption: [Hopskip Jump with different scales],
) <different_scale_experiment01>

@different_scale_experiment01 shows:
+ the attack doesn't manage to find the decision region for the blue class, since the blue points are still within the blue boundary.
+ The success rate on the overfit model is significantly lower (67%) compared to the normal model (100%).
+ The Wasserstein distance is larger for the normal class, indicating the points had to move further on average than for the overfit model.
+ Sum of squares is also larger for the normal model meaning the points had to move further than for the overfit model.

Separated by classes we obtain the following table:

#figure(
  table(
    columns: 3,
    [], [Normal], [Overfit],
    [Blue points], [6.308], [3.652],
    [Orange points], [3.855], [0.301],
  ),
  caption: [Different Scale By Class (Hopskip Jump)],
) <different_scale_table01>

@different_scale_table01 thoughts:
+ The Wasserstein distance is very small for the orange class and overfit model. The normal column is closer together but there is still a discrepancy.
+ Since the data here is the same as in @class_separated_table except for the variance of the blue class along x1 (x-axis), the wasserstein discrepancy within the normal model across the two classes is most likely due to this scaling effect. This would make sense since the scaling would mean more distance to make one distribution fit another.
+ *IDEA* - Scale the data by class across all features and see if this removes the scaling effect.


=== Separated by classes

From the example given in @fig01, if we split by classes we get the following:

#figure(
  table(
    columns: 3,
    [], [Normal], [Overfit],
    [Blue points], [2.959], [0.448],
    [Orange points], [2.786], [3.962],
  ),
  caption: [Separated by class experiment],
) <class_separated_table>

Some thoughts on @class_separated_table:
+ The normal column shows that the distances are roughly similar between both classes, whereas for the overfit model they vary widely.
+ For the overfit model, the blue points need to move significantly less than for the orange class, where they actually move quite a lot.
+ *IDEA*: There might be something here where we can split the distances by classes and see if the Wasserstein distance is similar.
  - This will most likely hold for SVMs since they try to find a boundary that separates the points. However, there will most likely be cases where this will fail, where you can keep the boundary the same but move the rest of the distribution for one class away from the boundary. However, this scenario can most likely be mitigated by setting the misclassification penalty to not be infinite.

=== Correlation between Wasserstein and Sum of Squares




