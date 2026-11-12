"""No new pass: per-story paired loss differences are not stored by the ledger, so only the pooled means
from the read calls are available. Collect them from the saved outputs of the eval scripts."""
print("stories 1-64   : base 1.64999  v1 1.65383 (+0.0038)  v2 1.65693 (+0.0069)")
print("stories 65-128 : base 1.51264                         v2 1.52046 (+0.0078)")
print("pooled (token-weighted approx, equal weight): v2 +0.0074")
