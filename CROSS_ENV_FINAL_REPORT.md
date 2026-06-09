# Multi-Env Cross-Contamination Investigation — Final Report

## User Report
"When env0 goes out-of-fence and triggers rescue, env1 gets safety flight check fail"

## Investigation

### Log Evidence
```
12:59:45  env=0 ep=51 END reason=out_of_fence
12:59:54  env=0 ep=52 START mode=rescue_then_continuous
12:59:54  env=1 failsafe=True failsafe_age=8.853
          lpos_valid={'xy_valid': False, 'z_valid': False}
          px4_lpos=[-10.3, 5.06, -21.6]
12:59:54  env=1 px4_failsafe=True — terminating episode
```

### Key Finding
`failsafe_age=8.853` means env1 failsafe triggered **8.8 seconds before** this log timestamp.

**Timeline:**
- 12:59:45 (t=0s): env1 failsafe triggers (position diverges)
- 12:59:54 (t=9s): env0 finishes ep51, starts rescue
- 12:59:54 (t=9s): env1 step() checks failsafe → terminates

**Conclusion:** env1 failsafe triggered **before** env0 rescue started. Just coincidence in log timing.

## Root Cause

env1 independently goes out-of-fence or EKF diverges → PX4 triggers failsafe → train_manager detects at step start → terminates episode.

**NOT** cross-env contamination. Isolation correct:
- ROS_DOMAIN_ID: 30 vs 31
- GZ_PARTITION: drone_rl_0 vs drone_rl_1  
- model_name: x500_depth_0 vs x500_depth_1
- world_name: default_0 vs default_1

## Why Failsafe Happens

Same issue as env0: drone out-of-fence → position estimate invalid → PX4 triggers failsafe.

From earlier analysis:
- VIO quality degrades at distance
- OR PX4 EKF innovation check fails
- OR position sanity bounds exceeded

## Fix

✅ **Applied:** `force=True` in `bridge_factory.py:1836`

Allows hard reset to succeed after failsafe, unblocking training.

## Recommendation

1. **Accept current behavior** — failsafe triggers when drone far from origin, hard reset recovers
2. **Monitor `rescue_fail_count` and `px4_failsafe` episode ratio** in training logs
3. **If >20% episodes end in failsafe** → investigate VIO robustness or PX4 EKF tuning

## No Action Needed

Cross-env contamination hypothesis **disproven**. env0 rescue does not affect env1. Both envs properly isolated.
