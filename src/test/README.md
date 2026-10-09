# NPI-M / A-NPI-M validation tests

## Suites and commands

- `NPISuite`: exact NPI-M hand-checkable frequencies, marginal bounds, distribution invariants, input validation, and regression rejection.
- `ANPISuite`: A-NPI-M hand-checkable frequencies, bounds, approximate-distribution invariants, permutation/symmetry behavior, and regression rejection.
- `NPIANPIComparisonSuite`: six representative cases plus exhaustive count vectors with 2–5 categories and counts in 0–5, excluding only the all-zero vectors because singleton probability bounds are undefined when the sample total is zero. It prints the evaluated case count, maximum marginal-bound difference, and empirical distribution/entropy disagreements.
- `CredalTreeComparisonSuite`: deterministic local Spark integration tests for a clear binary split, two symmetric positive-gain tied candidates, and a multiclass case with absent classes in branches. Console output summarizes singleton bounds, root impurity, root gain, selected split, tree structure and exact prediction vectors for both criteria.

Run all tests:

```bash
sbt test
```

Run individual suites:

```bash
sbt "testOnly org.apache.spark.mllib.tree.impurity.NPISuite"
sbt "testOnly org.apache.spark.mllib.tree.impurity.ANPISuite"
sbt "testOnly org.apache.spark.mllib.tree.impurity.NPIANPIComparisonSuite"
sbt "testOnly org.apache.spark.mllib.tree.impurity.CredalTreeComparisonSuite"
```

`Test / parallelExecution` is disabled because the integration suite owns one shared local `SparkSession` for its tests and stops it after the suite.

## Interpretation and numerical tolerances

`NPI-M and A-NPI-M are not assumed to be theoretically identical. The comparison tests verify the equality of singleton probability bounds and document the empirical agreement or disagreement of higher-level outputs for the evaluated cases.`

The marginal interval convention used here is $[\max(0,(c_i-1)/n),\min(1,(c_i+1)/n)]$ for each singleton category. The equality of those bounds is a construction-level property; it does **not** establish equality of credal sets, because exact NPI-M can impose additional joint constraints.

`1e-12` is used for arithmetic/formula comparisons and the exact NPI-M distribution. The A-NPI-M implementation stops its recursion once residual mass is below `0.001`, so its distribution-sum and permutation checks use `1e-3`, matching that production stopping criterion rather than hiding a model difference with broad rounding.

The comparison suite evaluates 9,320 nonzero integer frequency vectors (category counts 2–5, each class frequency 0–5). It asserts singleton-bound equality and each algorithm's probability/distribution invariants, and reports the count and maximum difference of empirical distribution and entropy disagreements. A separate fixed histogram example prints per-candidate gains and rankings calculated by each impurity. Distribution, entropy, tree gain, selected split, tree structure and predictions are empirical behavior, not general equivalence claims. The tests do not require higher-level agreement unless a named simple case has an independently hand-derived expected value.

The Spark integration suite records the root impurity and gain, actual root split, full debug tree structure, and predictions. The fixed-histogram comparison reports candidate-by-candidate gains and their induced ordering from the exposed criterion API; this validates empirical ranking on that fixture, but does not instrument the engine's internal candidate list. A matching split does not prove matching gain, tree structure, or predictions; each level is reported separately. Model predictions use the classifier's observed-count rule, not credal probability bounds.

The repository has no identified article-specific dataset, seed, or published numerical reference outputs. The checked-in SUSY data is project data, not documented as an article regression fixture; the automated suite therefore uses small deterministic local cases and makes no claim of reproducing a paper experiment. The exact implementation accepts only integer multinomial counts, so comparisons intentionally exclude fractional weighted counts. Zero-total impurity is tested separately; bounds and distributions are undefined for that case.
