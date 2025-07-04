# Manifests generation

In case of changes of the installed resources, it may be convenient to export
them from a running installation.

To do so, we are using the `eksporter` `krew` plugin which cleans out namespace-dependent
or status-related data to make the output more portable or reusable.

## `krew` installation
Refer to https://krew.sigs.k8s.io/docs/user-guide/setup/install/

## `eksporter` installation
Simply run
```
kubectl krew install eksporter
```

## `extract-manifests.sh` script
The `extract-manifests.sh` does the manifest extraction.


