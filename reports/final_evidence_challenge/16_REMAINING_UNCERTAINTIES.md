# REMAINING UNCERTAINTIES

There are zero remaining uncertainties regarding *how* the results were generated. The canonical configurations have been fully locked, verified, and computationally reproduced from scratch without relying on historical models.

The only remaining project engineering choices to acknowledge in the final paper/viva are:
1. Truncating the IIoT datasets chronologically at 7,000 samples.
2. The omission of `log1p` during preprocessing, causing degraded Standard MLP performance.
3. The synthetic nature of the DP privacy-utility demonstration.
