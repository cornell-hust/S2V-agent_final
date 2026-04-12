# TimeSearch-R Port Provenance

Source tree: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/TimeSearch-R`

Source commit: `e427ee64b54a0668520baa4086f7120fd63813f4`

Copied assets:

- `time_r1/environment/base.py`: abstract environment interface.
- `time_r1/utils/reward_utils.py`: conversational completion extraction helpers.
- `time_r1/eval/moment_retrieval_utils.py`: temporal IoU and average-precision helpers.

These files are kept under `third_party_ports/timesearch_r/` so v3 can depend on stable TimeSearch-R logic without importing the original checkout.
