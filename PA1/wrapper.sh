#!/bin/bash

LOCAL_RANK=$SLURM_LOCALID # for SLURM
CORE=$(($LOCAL_RANK % 14 * 2 + $LOCAL_RANK / 14))
echo "Process $LOCAL_RANK on $(hostname) bound to core $CORE"
exec numactl -C "$CORE" $@
