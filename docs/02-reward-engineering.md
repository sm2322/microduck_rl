# 02 — Reward engineering

A reward function is a specification. RL optimizes the letter of it, not the
behavior you meant. Most of this page is a catalogue of ways that goes wrong and
the rule that prevents it. They were all learned by watching a 25 cm robot do
something absurd.

## The sign convention (this one bit four envs)

There are two kinds of terms in `mdp.py`:

| kind | returns | weight sign | examples |
|---|---|---|---|
| task / tracking | `[0, 1]` | **positive** | `pose_target_match`, `flamingo_hold_score`, `track_linear_velocity` |
| mjlab cost | `>= 0` | **negative** | `action_rate_l2`, `body_ang_vel`, `self_collision_cost` |
| microduck penalty | `<= 0` (self-negating) | **positive** | `trunk_vertical_accel_penalty`, `pose_l1_penalty` |

A self-negating penalty with a *negative* weight double-negates into a reward
for the violation. The policy finds that immediately and farms it (butt-hopping,
crash-sits).

**The infallible check:** on every run, every `Episode_Reward/<penalty>` in
wandb must be `<= 0`. If it is positive, the weight or the function has the
wrong sign. Make this the first thing you look at.

## Define the behavior in hard gates, not gentle nudges

Each under-specified degree of freedom will be exploited:

- a "roll" becomes a ballistic whip,
- a "shoulder roll" goes sideways instead of sagittal,
- a "stand" becomes a head tripod,
- a "one-leg hold" becomes a hop or a lean on the lifted foot.

Encode what counts as the maneuver with **state-based gates**: required support
contact, orientation-axis checks, positive latches. A small penalty on the wrong
behavior loses to a large reward for a cheap approximation.

Flamingo does this with a multiplicative score gated by stance-foot contact:

```python
# src/mjlab_microduck/tasks/mdp.py
return pose_gaussian * height_gaussian * upright_gaussian * stance_foot_contact
```

Any partial pose — two feet down, hopping, trunk low — scores near zero.

## No jackpots

Any "reach X" reward that then pays per step is a jackpot: arriving early buys
arbitrary violence. Two rules:

- Rate-limit or slew the target. For commanded transitions, track a constant-rate
  blend; being ahead of the ramp pays zero, so slow *is* the optimum.
- Never gate a positive reward on a bad state ("reward for being low / fallen").
  The policy parks in the cheapest qualifying pose and farms it.

For progress, use potential-based shaping: pay Δprogress (e.g. `Δcos(tilt)`), so
rising pays, holding pays zero, and it cannot be farmed.

## Multiplicative beats additive at goal states

An additive stack of ten Gaussian terms has a compromise basin: get 80 % of
every term by leaning. A product collapses on any single deficient factor.

But pick the standard deviations so the **current** policy scores visibly. If
the product is `1e-30` at initialization, the gradient is invisible and nothing
happens. Start wide, verify `Episode_Reward/flamingo_hold` is non-zero in the
first 50 iterations, then tighten.

## Regularizers have two flavours

- **Motion blockers** (`body_ang_vel`, `angular_momentum`, tight pose std)
  penalize what a dynamic motion physically requires. Keep them low for
  dynamic tasks.
- **Smoothness** (`action_rate`, `joint_torque_rate`) damps jitter without
  blocking slow large motions. Safe to weight — but introduce *after* the skill
  exists. An attempt tax active during exploration makes "do nothing" win.

Compare **reward mass**, not weights, when copying a regularizer between envs.
PPO sees relative advantage: the same `action_rate` weight is 4× weaker under a
4×-larger task stack.

## Tracking standard deviations

The std is "the error you still care about", not "the maximum error". Too loose
has no gradient at small errors.

Before tightening, ask whether the error is escapable by the policy or inherent
to the behavior. A 38 %-of-body-mass head **must** oscillate while walking; a
tight instantaneous head-tracking std taxes walking so hard the policy stands
still. Price only the escapable part — e.g. L1 on a 1 s EMA charges DC bias and
lets the oscillation cancel.

## Commands and dead weights

- A command input that is never non-zero has dead weights forever. Every slot
  keeps a small non-zero sampling range from step 0, even at reward weight 0.
- **Zero-command behavior must be explicitly trained.** Uniform sampling almost
  never produces the all-zero command, which is exactly the deployment idle
  state. Use a `zero_command_prob`-style exact-zero bucket.
- Rare-but-important regions need explicit buckets (turn-in-place was 2 % of
  experience under independent uniform sampling and never trained).

## Joint limits

Joints park on hard limits because the stock `dof_pos_limits` cost only fires in
the last ~7.5 % of the range. Fix it with a qpos-side limit-proximity penalty on
the offending joints. Penalizing on the command side does not work: the wide
ctrlrange is intentional, low-kp servos need overshoot.

## Curricula

- Steps are env steps: `iteration × 24`.
- Use `microduck_mdp.reward_weight` for weight schedules and a dedicated
  params curriculum for command/event ranges. `reward_weight` is a step
  function, not an interpolation: discretize ramps into stages.
- Mutate term cfgs through the managers
  (`env.event_manager.get_term_cfg(...)`), never `env.cfg.events[...]` — managers
  deepcopy their cfg at init, so writes to `env.cfg` are silent no-ops.
- Phase-align every stage with what the policy has learned. If a wandb metric
  steps **down** exactly at a stage boundary, the pacing is wrong: stretch the
  stage or move the introduction later, never earlier.
- Reverse-curriculum spawns (start partway through the maneuver) are the
  reliable fix for "learns the start, never the last mile".

## The loop

1. Pick the closest existing task and copy its cfg.
2. Write down what "success" looks like as measurable states.
3. Implement one positive score and the hard gates first. No regularizers.
4. Smoke test. Check dimensions, NaN, every term computes.
5. Train a short run. Look at `Episode_Reward/<each term>`, not just the total.
6. Watch the video. Fix reward hacking one term at a time.
7. Add smoothness/regularization once the behavior exists.
8. Only then tune for transfer (DR stays inherited).

Expect two to five rounds of reward-hacking whack-a-mole. That is the job.
