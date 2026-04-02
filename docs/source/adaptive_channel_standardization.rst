Adaptive Channel Standardization for Clustering
==============================================

Purpose
-------
This document describes the canonical preprocessing path for clustering runs
that combine channels collected at different sampling intervals or sensor
resolutions.

The goal is to make cross-deployment and cross-dataset clustering more
comparable without switching to a learned or generative temporal upsampling
model. The default method is deterministic and physics-facing: smooth in time,
quantize when needed, smooth again, and compute derivatives in physical units.

Why This Exists
---------------
Mixed sensor streams often differ in two ways that matter for clustering:

- sampling interval / effective frequency
- value resolution / quantization sensitivity

If those channels are clustered directly, higher-rate or finer-resolution
channels can dominate derivative-based features and make the feature space
reflect sensor differences instead of behavioral differences.

The adaptive standardization path reduces that problem before feature
extraction.

Canonical Method
----------------
The initial supported method is ``adaptive_gaussian_quantized``.

For each configured source channel, the pipeline:

1. infers sampling interval from the median positive datetime difference
2. infers source resolution from the smallest nonzero value changes
3. chooses a smoothing window adaptively
4. smooths the source channel in seconds, not samples
5. optionally quantizes to a shared step
6. smooths again with the same time-based Gaussian window
7. computes derivatives using seconds as the denominator

The initial adaptive rule is:

- use ``base_smooth_seconds`` by default
- switch to ``coarse_smooth_seconds`` when either:
  - ``sampling_interval_s >= coarse_interval_threshold_s``
  - inferred resolution ``>= coarse_resolution_threshold``

Why Smoothing Is Defined In Seconds
-----------------------------------
The method must remain comparable across channels recorded at different
frequencies. A 6-sample smoothing window means very different things at
1 Hz, 5 Hz, and 20 Hz. A 6-second window does not.

This is why the clustering workflow converts smoothing durations to an
effective ``sigma`` in samples per deployment rather than storing sample-count
windows in the config.

Config Shape
------------
Standardization is opt-in per source channel via the run-level
``preprocessing.standardized_channels`` block.

Example:

.. code-block:: yaml

   preprocessing:
     standardized_channels:
       depth.depth:
         method: adaptive_gaussian_quantized
         quantize_step: 1.0
         base_smooth_seconds: 6
         coarse_smooth_seconds: 12
         coarse_interval_threshold_s: 5
         coarse_resolution_threshold: 1.0
         derivative_orders: [0, 1, 2]
         output_prefix: standardized_depth

This configuration creates synthetic clustering sources that can be referenced
from ``channel_feature_spec``:

.. code-block:: yaml

   channel_feature_spec:
     standardized_depth.depth_std:
       transforms: [raw]
       feature_set: full
     standardized_depth.depth_d1_std:
       transforms: [raw]
       feature_set: full
     standardized_depth.depth_d2_std:
       transforms: [raw]
       feature_set: full

Derived Outputs
---------------
For a source ``signal.channel`` and ``output_prefix`` value, the workflow
produces:

- order 0: ``output_prefix.channel_std``
- order 1: ``output_prefix.channel_d1_std``
- order 2: ``output_prefix.channel_d2_std``

These are synthetic clustering inputs. They are not written back into
``data.pkl`` signal_data. They exist only inside the clustering workflow.

Compatibility Validation
------------------------
If a channel is not standardized, the workflow checks whether its effective
sampling interval is compatible with the other active non-standardized
clustering inputs.

When enabled:

- compare median positive sampling intervals
- require relative deviation to remain within
  ``sampling_compatibility_tolerance``

If ``sampling_compatibility_tolerance`` is omitted, compatibility enforcement
is disabled for that run. This keeps legacy clustering runs unchanged unless
they explicitly opt into strict cross-channel sampling validation.

If incompatible non-standardized channels are mixed in one run, the workflow
fails fast with a clear error naming the offending channels and their inferred
sampling intervals/frequencies.

This is intentional. Silent mixing of incompatible raw channels is usually
worse than an early failure.

Worked Example: ``mile`` + ``mian`` Depth Clustering
----------------------------------------------------
The shared run ``mian_mile_depth_standardized_30s`` standardizes
``depth.depth`` for both datasets before clustering in 30-second bins.

The run:

- uses adaptive smoothing to account for different sensor cadence/resolution
- quantizes to a 1 m step
- computes standardized depth, first derivative, and second derivative
- clusters on those standardized depth-derived features instead of raw depth

Diagnostics
-----------
For each processed deployment, the workflow writes diagnostics containing:

- inferred sampling interval
- inferred sampling frequency
- inferred source resolution
- chosen smoothing window
- quantization step
- derived outputs produced
- compatibility validation results

It also writes a representative plot showing:

- raw source channel
- standardized channel
- first derivative
- second derivative

Default Recommendation
----------------------
- Use deterministic adaptive standardization as the canonical method for
  cross-sensor clustering.
- Only cluster raw non-standardized channels together when their sampling
  intervals are already compatible.
- Treat learned or generative temporal upsampling as experimental and optional,
  not as the default preprocessing path.
