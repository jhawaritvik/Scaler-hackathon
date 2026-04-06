"""Quick test to verify the fire simulation engine works."""

import sys
import os

# Add server directory to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

from terrain import generate_terrain, TASK_CONFIGS
from fire_simulation import FireSimulation

def test_easy():
    print("=" * 60)
    print("TESTING EASY TASK")
    print("=" * 60)

    config = TASK_CONFIGS["easy"]
    terrain = generate_terrain(config)

    print(f"\nTerrain generated:")
    print(f"  Grid: {config.grid_size}x{config.grid_size}")
    print(f"  Elevation range: {terrain.elevation.min()}-{terrain.elevation.max()}")
    fuel_counts = {i: int((terrain.fuel_type == i).sum()) for i in range(4)}
    print(f"  Fuel types: {fuel_counts}")
    print(f"  Water cells: {terrain.is_water.sum()}")
    print(f"  Structures: {len(terrain.structures)}")
    print(f"  Ignition points: {len(terrain.ignition_points)}")
    print(f"  Initial moisture range: {terrain.initial_fuel_moisture.min():.2f}-{terrain.initial_fuel_moisture.max():.2f}")

    # Print elevation map
    print(f"\nElevation map:")
    for r in range(config.grid_size):
        print("  " + " ".join(f"{terrain.elevation[r, c]}" for c in range(config.grid_size)))

    # Run simulation
    sim = FireSimulation(terrain)
    state = sim.reset()

    print(f"\n--- Step {state.step} ---")
    print(sim.get_grid_string())
    print(f"  Burning: {state.total_burning}, Burned: {state.total_burned}")
    print(f"  Temp: {state.temperature:.1f}°C, Humidity: {state.humidity:.0%}")
    print(f"  Wind: {state.wind_speed:.1f} km/h @ {state.wind_direction:.0f}°")

    # Run for all steps
    for i in range(config.max_steps):
        state = sim.tick()

        print(f"\n--- Step {state.step} ---")
        print(sim.get_grid_string())
        print(f"  Burning: {state.total_burning}, Burned: {state.total_burned}")
        print(f"  Temp: {state.temperature:.1f}°C, Humidity: {state.humidity:.0%}")
        print(f"  Wind: {state.wind_speed:.1f} km/h @ {state.wind_direction:.0f}°")
        dryness_index = state.temperature - (state.humidity * 100.0)
        print(f"  Dryness index: {dryness_index:.1f}")
        print(f"  Airflow potential peak: {state.airflow_potential.max():.2f}")
        print(f"  Structures - saved: {state.structures_saved}, lost: {state.structures_lost}")

        rewards = sim.compute_environmental_rewards()
        if rewards:
            total_r = sum(rewards.values())
            print(f"  Rewards: total={total_r:.3f} | {rewards}")

        if state.done:
            print(f"\n  >>> EPISODE DONE at step {state.step}")
            break

    print(f"\n{'=' * 60}")
    print(f"FINAL STATS")
    print(f"{'=' * 60}")
    print(f"  Total burned: {state.total_burned}")
    print(f"  Structures lost: {state.structures_lost}/{len(terrain.structures)}")
    print(f"  Steps completed: {state.step}")


if __name__ == "__main__":
    test_easy()
