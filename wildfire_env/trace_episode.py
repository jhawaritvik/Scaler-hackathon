"""Detailed episode trace showing exactly what happens at each step.

Exercises all 6 resource types and key mission types to verify the
complete dispatch → en_route → operating → returning lifecycle.
"""

import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from wildfire_env.models import GridPoint, ResourceAssignment, TargetSpec, WildfireAction
from wildfire_env.server.wildfire_env_environment import WildfireEnvironment


def _first_available(obs, resource_type: str):
    for u in obs.fleet_units:
        if u.resource_type == resource_type and u.status == "available":
            return u
    return None


def _print_step_result(obs, step_label: str):
    """Print compact result after a step."""
    print(f"{step_label} RESULT:")
    print(f"  Reward: {obs.reward:+.4f}")
    print(f"  Step: {obs.step}/{obs.max_steps}, Elapsed: {obs.elapsed_minutes} min")
    print(f"  Summary: {obs.last_action_summary}")
    if obs.last_action_error:
        print(f"  ERROR: {obs.last_action_error}")
    print(f"  Fire: burning={obs.burning_cells}, burned={obs.burned_cells}")
    print(f"  Weather: wind={obs.wind_speed} km/h @ {obs.wind_direction:.0f} deg, "
          f"temp={obs.temperature:.1f} C, humidity={obs.humidity:.0%}")
    print(f"  Active missions: {len(obs.active_missions)}")
    for m in obs.active_missions:
        print(f"    {m.unit_id}: status={m.status} mission={m.mission_type} "
              f"eta={m.eta_steps}s -- {m.summary}")
    print(f"  Available: {obs.resources_remaining}")
    print(f"  Structures: remaining={obs.structures_remaining}, lost={obs.structures_lost}")
    print(f"  Done: {obs.done}")
    print()
    print("GRID:")
    print(obs.grid)


def main():
    env = WildfireEnvironment()
    obs = env.reset()

    print("=" * 80)
    print(f"EPISODE START: task={obs.task_id}")
    print(f"Goal: {obs.goal}")
    print(f"Max steps: {obs.max_steps}, Step duration: {obs.step_minutes} min")
    print()
    print("GRID (step 0):")
    print(obs.grid)
    print()
    print(f"Fire: burning={obs.burning_cells}")
    print(f"Weather: wind={obs.wind_speed} km/h @ {obs.wind_direction:.0f} deg, "
          f"temp={obs.temperature} C, humidity={obs.humidity:.0%}")
    print(f"Structures:")
    for s in obs.structures:
        print(f"  {s.structure_id}: ({s.row},{s.col}) priority={s.priority} status={s.status}")
    print(f"Fire details:")
    for f in obs.fire_details:
        print(f"  ({f.row},{f.col}) intensity={f.intensity} timer={f.timer}")
    print()

    print("FLEET AT START:")
    print(f"  Totals: {obs.resource_totals}")
    print(f"  Available: {obs.resources_remaining}")
    for u in obs.fleet_units:
        print(f"  {u.unit_id}: type={u.resource_type} status={u.status} "
              f"pos=({u.current_row},{u.current_col})")
    print()

    # ──────────────────────────────────────────────
    # STEP 1: Crew + Engine + Smokejumper direct_attack
    # ──────────────────────────────────────────────
    fire = obs.fire_details[0] if obs.fire_details else None
    assignments = []

    if fire:
        fp = GridPoint(row=fire.row, col=fire.col)

        crew = _first_available(obs, "crews")
        if crew:
            assignments.append(ResourceAssignment(
                unit_id=crew.unit_id, mission_type="direct_attack",
                target=TargetSpec(target_kind="point", point=fp),
            ))
            print(f"STEP 1: {crew.unit_id} → direct_attack at ({fire.row},{fire.col})")

        engine = _first_available(obs, "engines")
        if engine:
            assignments.append(ResourceAssignment(
                unit_id=engine.unit_id, mission_type="direct_attack",
                target=TargetSpec(target_kind="point", point=fp),
            ))
            print(f"STEP 1: {engine.unit_id} → direct_attack at ({fire.row},{fire.col})")

        sj = _first_available(obs, "smokejumpers")
        if sj:
            assignments.append(ResourceAssignment(
                unit_id=sj.unit_id, mission_type="direct_attack",
                target=TargetSpec(target_kind="point", point=fp),
            ))
            print(f"STEP 1: {sj.unit_id} → direct_attack at ({fire.row},{fire.col})")

    print("=" * 80)
    obs = env.step(WildfireAction(assignments=assignments))
    _print_step_result(obs, "STEP 1")

    # ──────────────────────────────────────────────
    # STEP 2: Helicopter water_drop + Air tanker retardant_drop
    # ──────────────────────────────────────────────
    assignments = []
    target_point = None
    if obs.fire_details:
        f2 = obs.fire_details[0]
        target_point = GridPoint(row=f2.row, col=f2.col)
    elif obs.heat_warnings:
        hw = obs.heat_warnings[0]
        target_point = GridPoint(row=hw.row, col=hw.col)

    heli = _first_available(obs, "helicopters")
    if heli and target_point:
        assignments.append(ResourceAssignment(
            unit_id=heli.unit_id, mission_type="water_drop",
            target=TargetSpec(target_kind="area", center=target_point, radius=1),
            drop_configuration="salvo",
        ))
        print(f"STEP 2: {heli.unit_id} → water_drop at ({target_point.row},{target_point.col})")

    tanker = _first_available(obs, "airtankers")
    if tanker and target_point:
        assignments.append(ResourceAssignment(
            unit_id=tanker.unit_id, mission_type="retardant_drop",
            target=TargetSpec(target_kind="area", center=target_point, radius=1),
            drop_configuration="trail",
        ))
        print(f"STEP 2: {tanker.unit_id} → retardant_drop (trail) at ({target_point.row},{target_point.col})")

    # Dozer line_construction ahead of fire
    dozer = _first_available(obs, "dozers")
    if dozer and target_point:
        line_row = max(0, target_point.row - 3)
        assignments.append(ResourceAssignment(
            unit_id=dozer.unit_id, mission_type="line_construction",
            target=TargetSpec(
                target_kind="line",
                waypoints=[
                    GridPoint(row=line_row, col=max(0, target_point.col - 2)),
                    GridPoint(row=line_row, col=min(14, target_point.col + 2)),
                ],
            ),
        ))
        print(f"STEP 2: {dozer.unit_id} → line_construction at row {line_row}")

    print("=" * 80)
    obs = env.step(WildfireAction(assignments=assignments))
    _print_step_result(obs, "STEP 2")

    # ──────────────────────────────────────────────
    # STEP 3: Engine wet_line
    # ──────────────────────────────────────────────
    assignments = []
    engine2 = _first_available(obs, "engines")
    if engine2 and target_point:
        wl_row = max(0, target_point.row - 2)
        assignments.append(ResourceAssignment(
            unit_id=engine2.unit_id, mission_type="wet_line",
            target=TargetSpec(
                target_kind="line",
                waypoints=[
                    GridPoint(row=wl_row, col=max(0, target_point.col - 1)),
                    GridPoint(row=wl_row, col=min(14, target_point.col + 1)),
                ],
            ),
        ))
        print(f"STEP 3: {engine2.unit_id} → wet_line at row {wl_row}")

    print("=" * 80)
    obs = env.step(WildfireAction(assignments=assignments))
    _print_step_result(obs, "STEP 3")

    # ──────────────────────────────────────────────
    # STEPS 4+: Observe and let missions complete
    # ──────────────────────────────────────────────
    for step_num in range(4, min(obs.max_steps + 1, 12)):
        if obs.done:
            break
        print()
        print("=" * 80)
        print(f"STEP {step_num}: No new assignments (observe)")
        obs = env.step(WildfireAction())
        _print_step_result(obs, f"STEP {step_num}")

    print()
    print("=" * 80)
    print("EPISODE SUMMARY")
    print(f"  Task: {obs.task_id}")
    print(f"  Final step: {obs.step}/{obs.max_steps}")
    print(f"  Done: {obs.done}")
    state = env.state
    print(f"  Total reward: {state.total_reward}")
    print(f"  Structures lost: {obs.structures_lost}")
    print(f"  Burned cells: {obs.burned_cells}")


if __name__ == "__main__":
    main()
