# Threshold overrides

`--threshold_overrides_file` takes in a yaml file with error thresholds specified for specific backend and platform configuration. Currently, two types of error overrides are allowed: maximum error and near zero.

For maximum error, a blanket `max_error` is specified to override the parent classes relative error threshold.

For near zero override, `ignore_near_zero_errors` is specified to allow some fields to pass with higher relative error if the absolute error is very small. Additionally, it is also possible to define a global near zero value for all remaining fields not specified in `ignore_near_zero_errors`. This is done by specifying `all_other_near_zero=<value>`.

Override yaml file should have one of the following formats:

## One near zero value for all variables

```Stencil_name:
 - backend: <backend>
   max_error: <value>
   near_zero: <value>
   ignore_near_zero_errors:
    - <var1>
    - <var2>
    - ...
```
## Variable specific near zero value

```Stencil_name:
 - backend: <backend>
   max_error: <value>
   ignore_near_zero_errors:
    <var1>:<value1>
    <var2>:<value2>
    ...
```

## [optional] Global near zero value for remaining fields

```Stencil_name:
 - backend: <backend>
   max_error: <value>
   ignore_near_zero_errors:
    <var1>:<value1>
    <var2>:<value2>
   all_other_near_zero:<global_value>
    ...
```

where fields other than `var1` and `var2` will use `global_value`.

## [optional] Deactivate fused multiply-add (FMA) on CUDA device
Due to potential arithmetic difference between x86-64 and CUDA architecture on
multiply-add operations, you can deactivate it on CUDA. FMA are not _incorrect_ and
are faster to compute and therefore should be retain for production code.

WARNING: This will require a recompile, make sure caches are not already set.

```Stencil_name:
 - backend: <backend>
   no_cuda_fms: true
    ...
```
