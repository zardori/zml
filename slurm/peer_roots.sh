#!/bin/bash
# Builds ZML_PEER_ROOTS: the repo roots of all project members on this cluster, colon-separated.
#
# Each member works in their own scratch dir, so config inputs (precomputed latents, LoRA
# checkpoints) produced by one member are missing from another's repo. The dirs are group-readable,
# so the entrypoints (zml/paths.py) fall back to searching these roots for any repo-relative input
# path that is absent locally.
#
# Sourced with ZML_CLUSTER set (by slurm/{athena,helios}.sh on a compute node, by
# slurm/check_config_paths.sh on a login node). In the per-cluster templates below {user} expands
# to the plgrid login and {name} to the login without its "plg" prefix.

ZML_PEER_USERS=(plgzardori plgpoblos plgbtcaf)

case "${ZML_CLUSTER:?peer_roots.sh: ZML_CLUSTER must be set (athena|helios)}" in
    athena)
        if [ -z "${PLG_GROUPS_STORAGE:-}" ]; then
            echo "WARNING: PLG_GROUPS_STORAGE unset, peer repo roots will be wrong." >&2
        fi
        ZML_PEER_ROOT_TEMPLATE="${PLG_GROUPS_STORAGE:-}/plggtriplane/{name}/zml"
        ;;
    helios)
        ZML_PEER_ROOT_TEMPLATE="/net/scratch/hscra/plgrid/{user}/zml"
        ;;
    *)
        echo "WARNING: peer_roots.sh: unknown cluster '$ZML_CLUSTER', no peer roots." >&2
        ZML_PEER_ROOT_TEMPLATE=""
        ;;
esac

_zml_roots=()
for _zml_user in "${ZML_PEER_USERS[@]}"; do
    [ -n "$ZML_PEER_ROOT_TEMPLATE" ] || continue
    _zml_root="${ZML_PEER_ROOT_TEMPLATE//\{user\}/$_zml_user}"
    _zml_roots+=("${_zml_root//\{name\}/${_zml_user#plg}}")
done
export ZML_PEER_ROOTS=$(IFS=:; echo "${_zml_roots[*]:-}")
unset _zml_roots _zml_user _zml_root
