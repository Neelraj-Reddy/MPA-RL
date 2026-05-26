"""
Fuel Consumption Analysis

Analyzes fuel burn rates under different operational scenarios to understand
why boats are running out of fuel despite 12,000L capacity.
"""

import numpy as np

def calculate_fuel_burn(speed, cargo_ratio, net_deployed, net_type=1):
    """
    Calculate fuel consumption per hour based on operational state
    
    Parameters from fleet_physics.py:
    - idle_fuel_cost = 8.0 L/hr (base systems)
    - hull_drag_empty = 1.5
    - hull_drag_full = 3.5
    - net_drag_coeffs = [2.0, 8.0] (light, heavy)
    - wind_drag_coeff = 0.5
    - max_speed = 2.0 units/hr
    """
    
    # Constants from fleet_physics.py
    idle_fuel_cost = 8.0
    hull_drag_empty = 1.5
    hull_drag_full = 3.5
    net_drag_coeffs = [2.0, 8.0]
    wind_drag_coeff = 0.5
    
    # Assume water current and wind are negligible (worst case, boat speed = water speed)
    speed_thru_water = speed
    speed_thru_air = speed
    
    # Dynamic hull drag based on cargo
    dynamic_hull_drag = hull_drag_empty + (hull_drag_full - hull_drag_empty) * cargo_ratio
    temp_modifier = 1.0  # Assume normal temperature
    
    # Calculate each component
    water_resistance = (dynamic_hull_drag * temp_modifier) * (speed_thru_water ** 3)
    
    cargo_weight_factor = cargo_ratio * 50.0
    cargo_drag = cargo_weight_factor * (speed_thru_water ** 2) * 0.5
    
    net_resistance = 0.0
    if net_deployed:
        net_resistance = net_drag_coeffs[net_type] * (speed_thru_water ** 2)
    
    wind_resistance = wind_drag_coeff * (speed_thru_air ** 2)
    
    total_fuel_burn = idle_fuel_cost + water_resistance + cargo_drag + net_resistance + wind_resistance
    
    return {
        'total': total_fuel_burn,
        'idle': idle_fuel_cost,
        'water_resistance': water_resistance,
        'cargo_drag': cargo_drag,
        'net_resistance': net_resistance,
        'wind_resistance': wind_resistance
    }


def print_scenario(scenario_name, speed, cargo_ratio, net_deployed, net_type=1):
    """Print fuel consumption for a scenario"""
    fuel = calculate_fuel_burn(speed, cargo_ratio, net_deployed, net_type)
    
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")
    print(f"Speed: {speed:.2f} units/hr")
    print(f"Cargo: {cargo_ratio*100:.0f}%")
    print(f"Net: {'Deployed (' + ['Light', 'Heavy'][net_type] + ')' if net_deployed else 'Stowed'}")
    print(f"\nFuel Consumption Breakdown:")
    print(f"  Idle (systems):      {fuel['idle']:>8.2f} L/hr")
    print(f"  Water resistance:    {fuel['water_resistance']:>8.2f} L/hr")
    print(f"  Cargo drag:          {fuel['cargo_drag']:>8.2f} L/hr")
    print(f"  Net resistance:      {fuel['net_resistance']:>8.2f} L/hr")
    print(f"  Wind resistance:     {fuel['wind_resistance']:>8.2f} L/hr")
    print(f"  {'-'*50}")
    print(f"  TOTAL:               {fuel['total']:>8.2f} L/hr")
    
    # Calculate operational time with 12,000L tank
    max_fuel = 12000.0
    hours_of_operation = max_fuel / fuel['total']
    days_of_operation = hours_of_operation / 24.0
    
    print(f"\nOperational Time (with 12,000L tank):")
    print(f"  {hours_of_operation:.1f} hours = {days_of_operation:.1f} days")
    
    return fuel['total']


def analyze_typical_trip():
    """Analyze a typical fishing trip"""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*18 + "FUEL CONSUMPTION ANALYSIS" + " "*25 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Scenario 1: Docked at port (idle)
    burn_idle = print_scenario("Docked at Port (Idle)", speed=0.0, cargo_ratio=0.0, net_deployed=False)
    
    # Scenario 2: Cruising to fishing grounds (empty)
    burn_cruise_empty = print_scenario("Cruising to Fishing Grounds", speed=2.0, cargo_ratio=0.0, net_deployed=False)
    
    # Scenario 3: Active fishing (slow speed, net deployed, light net)
    burn_fishing_light = print_scenario("Active Fishing (Light Net)", speed=1.0, cargo_ratio=0.2, net_deployed=True, net_type=0)
    
    # Scenario 4: Active fishing (slow speed, net deployed, heavy net)
    burn_fishing_heavy = print_scenario("Active Fishing (Heavy Net)", speed=1.0, cargo_ratio=0.2, net_deployed=True, net_type=1)
    
    # Scenario 5: Returning home with full cargo
    burn_return_full = print_scenario("Returning to Port (Full Cargo)", speed=2.0, cargo_ratio=1.0, net_deployed=False)
    
    # Scenario 6: Maximum burn (full speed, full cargo, heavy net)
    burn_max = print_scenario("MAXIMUM BURN (Full Speed + Heavy Net + Full Cargo)", speed=2.0, cargo_ratio=1.0, net_deployed=True, net_type=1)
    
    # Calculate a realistic trip profile
    print("\n" + "="*70)
    print("REALISTIC TRIP PROFILE")
    print("="*70)
    
    # Assume a 100km round trip (50km each way)
    # At speed 2.0, that's 25 hours each way = 50 hours travel time
    # Assume 50 hours of active fishing
    # Total trip = 100 hours
    
    travel_hours = 50  # Round trip at max speed
    fishing_hours = 50  # Active fishing time
    
    fuel_for_travel = (travel_hours / 2) * burn_cruise_empty + (travel_hours / 2) * burn_return_full
    fuel_for_fishing = fishing_hours * ((burn_fishing_light + burn_fishing_heavy) / 2)  # Mix of light and heavy nets
    
    total_fuel_needed = fuel_for_travel + fuel_for_fishing
    
    print(f"\nTrip duration: {travel_hours + fishing_hours} hours ({(travel_hours + fishing_hours)/24:.1f} days)")
    print(f"  Travel time: {travel_hours} hours ({travel_hours/24:.1f} days)")
    print(f"  Fishing time: {fishing_hours} hours ({fishing_hours/24:.1f} days)")
    print(f"\nFuel required:")
    print(f"  Travel fuel:  {fuel_for_travel:>8.1f} L")
    print(f"  Fishing fuel: {fuel_for_fishing:>8.1f} L")
    print(f"  {'-'*50}")
    print(f"  TOTAL:        {total_fuel_needed:>8.1f} L")
    print(f"\nTank capacity: 12,000 L")
    print(f"Fuel margin:   {12000 - total_fuel_needed:>8.1f} L ({(12000 - total_fuel_needed)/12000*100:.1f}%)")
    
    if total_fuel_needed > 12000:
        print(f"\n⚠️  WARNING: Trip requires MORE fuel than tank capacity!")
        print(f"   Deficit: {total_fuel_needed - 12000:.1f} L")
    else:
        print(f"\n✓ Trip is feasible with current fuel capacity")
    
    # Calculate how many trips before running dry
    print("\n" + "="*70)
    print("DEPLOYMENT ANALYSIS")
    print("="*70)
    
    # In deployment, boats refuel at port each time they return
    # So the question is: can they make ONE round trip?
    
    trips_possible = 12000 / total_fuel_needed
    print(f"Trips possible per refuel: {trips_possible:.2f}")
    
    if trips_possible < 1.0:
        print(f"\n❌ CRITICAL ISSUE: Boats cannot complete even ONE trip!")
        print(f"   They will run out of fuel {(1.0 - trips_possible)*100:.0f}% through the trip")
    else:
        print(f"\n✓ Boats can complete {int(trips_possible)} full trip(s) per refuel")
    
    # Now let's see what happens if boats are fishing continuously without returning
    print("\n" + "="*70)
    print("CONTINUOUS OPERATION (No Refueling)")
    print("="*70)
    
    # If boats just fish non-stop at fishing grounds
    continuous_fishing_hours = 12000 / burn_fishing_heavy
    print(f"Continuous fishing time: {continuous_fishing_hours:.1f} hours = {continuous_fishing_hours/24:.1f} days")
    
    # If boats are idle/drifting
    idle_hours = 12000 / burn_idle
    print(f"Idle/drifting time:      {idle_hours:.1f} hours = {idle_hours/24:.1f} days")


def check_deployment_log():
    """Analyze actual deployment results"""
    print("\n" + "="*70)
    print("DEPLOYMENT LOG ANALYSIS")
    print("="*70)
    
    # From previous deployment logs, we saw boats running out of fuel around day 90
    # With 12,000L capacity
    
    print("\nFrom deployment_log.txt observations:")
    print("  - Boats running out of fuel by day 90")
    print("  - 14/15 boats dead by that point")
    print("  - Max fuel: 12,000L, Idle cost: 8 L/hr")
    print("\nCalculations:")
    print(f"  Pure idle time: 12,000L ÷ 8 L/hr = {12000/8:.0f} hours = {12000/8/24:.0f} days")
    print("\nThis means boats are burning fuel MUCH faster than idle rate.")
    print("Likely causes:")
    print("  1. Active fishing (net deployed at speed)")
    print("  2. High-speed chasing of fish schools")
    print("  3. Heavy cargo + high speed combinations")
    print("  4. Boats not returning to port frequently enough")


if __name__ == "__main__":
    analyze_typical_trip()
    check_deployment_log()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("\n1. AGENT TRAINING: Agents need to learn fuel-efficient behaviors:")
    print("   - Return to port more frequently")
    print("   - Fish at slower speeds")
    print("   - Avoid depleting nets at high speed")
    print("\n2. PHYSICS ADJUSTMENTS: Consider reducing drag coefficients:")
    print("   - Current net_drag_coeffs = [2.0, 8.0]")
    print("   - Current hull_drag_full = 3.5")
    print("\n3. FUEL CAPACITY: Current 12,000L may be insufficient for:")
    print("   - Long-range fishing (>50 units from port)")
    print("   - Extended fishing sessions (>50 hours)")
    print("\n4. REWARD STRUCTURE: Enhance fuel-awareness penalties:")
    print("   - Heavier penalty for fuel death (-200 implemented)")
    print("   - Bonus for fuel-efficient catches")
    print("   - Progressive penalty as fuel drops below 50%")
    print("="*70)
