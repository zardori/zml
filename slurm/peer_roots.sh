#!/bin/bash
# Builds ZML_PEER_ROOTS: the repo roots of all project members on this cluster, colon-separated.
#
# Each member works in their own scratch dir, so config inputs (precomputed latents, LoRA
# checkpoints) produced by one member are missing from another's repo. The dirs are group-readable,
# so the entrypoints (zml/paths.py) fall back to searching these roots for any repo-relative input
# path that is absent locally.
#
# Sourced by slurm/{athena,helios}.sh with ZML_PEER_ROOT_TEMPLATE set to that cluster's layout;
# {user} expands to the plgrid login, {name} to the login without its "plg" prefix.

ZML_PEER_USERS=(plgzardori plgpoblos plgbtcaf)

_zml_roots=()
for _zml_user in "${ZML_PEER_USERS[@]}"; do
    _zml_root="${ZML_PEER_ROOT_TEMPLATE//\{user\}/$_zml_user}"
    _zml_roots+=("${_zml_root//\{name\}/${_zml_user#plg}}")
done
export ZML_PEER_ROOTS=$(IFS=:; echo "${_zml_roots[*]}")
unset _zml_roots _zml_user _zml_root
