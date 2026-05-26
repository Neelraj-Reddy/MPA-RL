import json
import time
import numpy as np
import streamlit as st

from heuristic_simulator import HeuristicSimulation


st.set_page_config(
    page_title='Fleet Simulation Dashboard',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.title('Fishing Fleet Simulation Dashboard')
st.caption('Training-free fleet simulation with configurable days, GIF range, smoothness, and port-level deployment control.')


def _downsample_xy(x_values, y_values, max_points):
    n = len(x_values)
    if n <= max_points:
        return x_values, y_values
    stride = max(1, n // max_points)
    return x_values[::stride], y_values[::stride]


if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'media_nonce' not in st.session_state:
    st.session_state['media_nonce'] = 0

with st.sidebar:
    view_mode = st.radio('View', options=['Batch Simulation', 'Frame by Frame View'], index=0)
    st.header('Simulation Setup')

    simulation_days = st.slider('Run Duration (days)', min_value=1, max_value=1825, value=365, step=1)
    env_width = st.slider('Environment Width', min_value=60, max_value=180, value=100, step=10)
    env_height = st.slider('Environment Height', min_value=60, max_value=180, value=100, step=10)
    initial_fish = st.slider('Initial Fish Schools', min_value=500, max_value=8000, value=2500, step=100)

    st.header('Ports and Boats')
    num_ports = st.slider('Number of Ports', min_value=1, max_value=8, value=5, step=1)

    default_boats = []
    for p in range(num_ports):
        default_val = 3 if p < 5 else 0
        boats = st.number_input(
            f'Boats at Port {p}',
            min_value=0,
            max_value=60,
            value=default_val,
            step=1,
            key=f'boats_port_{p}',
        )
        default_boats.append(int(boats))

    configured_boats = int(sum(default_boats))
    use_custom_ports = st.toggle('Set Custom Port Locations', value=False)
    custom_port_locations = None
    if use_custom_ports:
        custom_port_locations = []
        st.caption('Set initial port coordinates (x, y).')
        for p in range(num_ports):
            pc1, pc2 = st.columns(2)
            px = pc1.number_input(
                f'Port {p} X',
                min_value=0.0,
                max_value=float(env_width - 1),
                value=float((p + 1) * env_width / (num_ports + 1)),
                step=1.0,
                key=f'port_x_{p}',
            )
            py = pc2.number_input(
                f'Port {p} Y',
                min_value=0.0,
                max_value=float(env_height - 1),
                value=float(env_height * 0.5),
                step=1.0,
                key=f'port_y_{p}',
            )
            custom_port_locations.append([float(px), float(py)])

    highlight_options = [-1] + list(range(configured_boats))
    highlight_boat = st.selectbox(
        'Highlight Specific Boat',
        options=highlight_options,
        format_func=lambda x: 'None' if x == -1 else f'Boat {x}',
    )

    focus_port = st.selectbox(
        'Focus Port in GIF (highlight boats from this port)',
        options=[-1] + list(range(num_ports)),
        format_func=lambda x: 'All Ports' if x == -1 else f'Port {x}',
    )

    st.header('GIF Controls')
    max_days = float(simulation_days)
    capture_start_day = st.slider('GIF Start Day', min_value=0.0, max_value=max_days, value=0.0, step=1.0)
    capture_end_day = st.slider('GIF End Day', min_value=0.0, max_value=max_days, value=max_days, step=1.0)
    target_frames = st.slider('Target Captured Frames', min_value=80, max_value=1200, value=420, step=20)
    gif_fps = st.slider('GIF FPS (higher = smoother)', min_value=4, max_value=30, value=12, step=1)
    media_format = st.selectbox('Media Output', options=['mp4', 'gif'], index=0)

    st.header('Performance')
    max_steps = max(1, simulation_days * 24)
    default_metric_step = max(1, int(max_steps // 3000))
    metrics_interval_steps = st.slider('Metrics Sampling Interval (steps)', min_value=1, max_value=96, value=default_metric_step, step=1)
    chart_max_points = st.slider('Max Chart Points', min_value=300, max_value=5000, value=1500, step=100)

    st.header('Ecology and Policy')
    mpa_enabled = st.toggle('Enable MPA', value=True)
    mpa_start_day = st.slider('MPA Start Day', min_value=0.0, max_value=max_days, value=0.0, step=1.0)
    mpa_end_day = st.slider('MPA End Day', min_value=0.0, max_value=max_days, value=max_days, step=1.0)

    random_seed = st.number_input('Random Seed (for reproducibility)', min_value=0, max_value=1000000, value=42, step=1)

    run_btn = st.button('Run Simulation', type='primary', use_container_width=True)

if view_mode == 'Frame by Frame View':
    st.subheader('Frame by Frame View - 3 Scenario Comparison')
    st.caption('Compare Normal (boats + MPA) vs No Boats vs No MPA. All advance together; only visible view renders.')

    init_col, prev_col, next_col, reset_col = st.columns([2, 1, 1, 1])
    init_clicked = init_col.button('Initialize Session', type='primary', use_container_width=True)
    prev_clicked = prev_col.button('Previous', use_container_width=True)
    next_clicked = next_col.button('Next', use_container_width=True)
    reset_clicked = reset_col.button('Reset', use_container_width=True)

    if reset_clicked:
        st.session_state.pop('ff_sims', None)
        st.session_state.pop('ff_histories', None)

    if init_clicked:
        # Initialize 3 parallel simulations
        ff_normal = HeuristicSimulation(
            env_width=env_width,
            env_height=env_height,
            hours_per_tick=1,
            initial_fish=initial_fish,
            num_ports=num_ports,
            boats_per_port=default_boats,
            port_locations=custom_port_locations,
            simulation_days=simulation_days,
            mpa_enabled=mpa_enabled,
            seed=int(random_seed),
        )
        ff_normal.start_interactive(
            mpa_start_day=float(mpa_start_day),
            mpa_end_day=float(mpa_end_day),
        )
        
        ff_no_boats = HeuristicSimulation(
            env_width=env_width,
            env_height=env_height,
            hours_per_tick=1,
            initial_fish=initial_fish,
            num_ports=num_ports,
            boats_per_port=[0] * num_ports,  # No boats
            port_locations=custom_port_locations,
            simulation_days=simulation_days,
            mpa_enabled=mpa_enabled,
            seed=int(random_seed),
        )
        ff_no_boats.start_interactive(
            mpa_start_day=float(mpa_start_day),
            mpa_end_day=float(mpa_end_day),
        )
        
        ff_no_mpa = HeuristicSimulation(
            env_width=env_width,
            env_height=env_height,
            hours_per_tick=1,
            initial_fish=initial_fish,
            num_ports=num_ports,
            boats_per_port=default_boats,
            port_locations=custom_port_locations,
            simulation_days=simulation_days,
            mpa_enabled=False,  # No MPA
            seed=int(random_seed),
        )
        ff_no_mpa.start_interactive(
            mpa_start_day=0,
            mpa_end_day=0,
        )
        
        st.session_state['ff_sims'] = {
            'normal': ff_normal,
            'no_boats': ff_no_boats,
            'no_mpa': ff_no_mpa,
        }
        st.session_state['ff_histories'] = {
            'normal': [ff_normal.get_interactive_snapshot()],
            'no_boats': [ff_no_boats.get_interactive_snapshot()],
            'no_mpa': [ff_no_mpa.get_interactive_snapshot()],
        }

    if 'ff_sims' not in st.session_state:
        st.info('Click Initialize Session to start frame-by-frame stepping.')
        st.stop()

    ff_sims = st.session_state['ff_sims']
    ff_histories = st.session_state['ff_histories']

    jump_col1, jump_col2, jump_col3 = st.columns([2, 1, 1])
    jump_steps = jump_col1.number_input('Jump Steps', min_value=1, max_value=5000, value=100, step=1, key='ff_jump_steps')
    jump_back_clicked = jump_col2.button('Back N', use_container_width=True)
    jump_fwd_clicked = jump_col3.button('Forward N', use_container_width=True)

    if prev_clicked:
        for scenario_key in ['normal', 'no_boats', 'no_mpa']:
            if len(ff_histories[scenario_key]) > 1:
                ff_histories[scenario_key].pop()
                ff_sims[scenario_key].load_interactive_snapshot(ff_histories[scenario_key][-1])

    if jump_back_clicked:
        for scenario_key in ['normal', 'no_boats', 'no_mpa']:
            if len(ff_histories[scenario_key]) > 1:
                n = int(min(jump_steps, len(ff_histories[scenario_key]) - 1))
                for _ in range(n):
                    ff_histories[scenario_key].pop()
                ff_sims[scenario_key].load_interactive_snapshot(ff_histories[scenario_key][-1])

    if next_clicked:
        for scenario_key in ['normal', 'no_boats', 'no_mpa']:
            advanced = ff_sims[scenario_key].interactive_next_step()
            if advanced:
                ff_histories[scenario_key].append(ff_sims[scenario_key].get_interactive_snapshot())

    if jump_fwd_clicked:
        n = int(jump_steps)
        with st.spinner(f'Advancing {n} steps...'):
            for _ in range(n):
                any_advanced = False
                for scenario_key in ['normal', 'no_boats', 'no_mpa']:
                    advanced = ff_sims[scenario_key].interactive_next_step()
                    if advanced:
                        ff_histories[scenario_key].append(ff_sims[scenario_key].get_interactive_snapshot())
                        any_advanced = True
                if not any_advanced:
                    break

    current_step = int(ff_sims['normal'].interactive_step)
    total_steps = int(ff_sims['normal'].total_steps)
    pct = int((current_step / max(1, total_steps)) * 100)
    st.progress(min(100, pct))
    st.write(f'Step {current_step}/{total_steps} ({pct}%)')

    # Tabs for 3 scenarios
    tab_normal, tab_no_boats, tab_no_mpa = st.tabs(['Normal (Boats + MPA)', 'No Boats', 'No MPA'])

    # Helper function to display each scenario
    def display_scenario(scenario_key, tab_ui, ff_sim, ff_history, show_boat_highlight=True):
        with tab_ui:
            if show_boat_highlight and ff_sim.total_boats > 0:
                ff_highlight_boat = st.selectbox(
                    f'Highlight Boat ({scenario_key})',
                    options=[-1] + list(range(ff_sim.total_boats)),
                    format_func=lambda x: 'None' if x == -1 else f'Boat {x}',
                    key=f'ff_highlight_boat_{scenario_key}',
                )
            else:
                ff_highlight_boat = -1

            focus_mask = None
            if focus_port != -1 and show_boat_highlight and ff_sim.total_boats > 0:
                focus_mask = [p == int(focus_port) for p in ff_sim.boat_home_port_index]

            frame = ff_sim._render_frame(
                max(0, current_step),
                np.array(focus_mask, dtype=bool) if focus_mask is not None else None,
                highlight_boat=None if ff_highlight_boat == -1 else int(ff_highlight_boat),
            )
            st.image(frame, caption=f'{scenario_key} - Step {current_step}')

            if ff_sim.total_boats > 0:
                fish_biomass = float(ff_sim.fish.energies.sum()) if ff_sim.fish.num_schools > 0 else 0.0
                plankton_total = float(ff_sim.env.plankton_grid.sum())
                total_energy = fish_biomass + plankton_total
                total_biomass = fish_biomass
                avg_school_energy = float(ff_sim.fish.energies.mean()) if ff_sim.fish.num_schools > 0 else 0.0
                captured_biomass = float(ff_sim.boat_catch.sum() + ff_sim.fleet.cargo_levels.sum())
                
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric('Fish Schools', int(ff_sim.fish.num_schools))
                c2.metric('Avg School Energy', f'{avg_school_energy:.2f}')
                c3.metric('Avg Fleet Fuel', f'{float(ff_sim.fleet.fuel_levels.mean()):.1f}')
                c4.metric('Total Energy', f'{total_energy:.1f}')
                c5.metric('Total Biomass', f'{total_biomass:.1f}')
                c6.metric('Captured Biomass', f'{captured_biomass:.1f}t')

                selected_boat_ff = st.selectbox(
                    f'Choose Boat ({scenario_key})',
                    options=list(range(ff_sim.total_boats)),
                    format_func=lambda b: f'Boat {b} (Port {ff_sim.boat_home_port_index[b]})',
                    key=f'ff_selected_boat_{scenario_key}',
                )

                ff_steps = [snap['interactive_step'] for snap in ff_history]
                ff_boat_fuel = [float(snap['fleet']['fuel_levels'][selected_boat_ff]) for snap in ff_history]
                ff_boat_cargo = [float(snap['fleet']['cargo_levels'][selected_boat_ff]) for snap in ff_history]
                ff_boat_catch = [float(snap['boat_catch'][selected_boat_ff]) for snap in ff_history]

                ff_chart = {
                    'step': ff_steps,
                    'boat_fuel': ff_boat_fuel,
                    'boat_cargo': ff_boat_cargo,
                    'boat_cumulative_catch': ff_boat_catch,
                }
                st.line_chart(ff_chart, x='step')
            else:
                fish_biomass = float(ff_sim.fish.energies.sum()) if ff_sim.fish.num_schools > 0 else 0.0
                plankton_total = float(ff_sim.env.plankton_grid.sum())
                total_energy = fish_biomass + plankton_total
                total_biomass = fish_biomass
                avg_school_energy = float(ff_sim.fish.energies.mean()) if ff_sim.fish.num_schools > 0 else 0.0
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Fish Schools', int(ff_sim.fish.num_schools))
                c2.metric('Avg School Energy', f'{avg_school_energy:.2f}')
                c3.metric('Total Energy', f'{total_energy:.1f}')
                c4.metric('Total Biomass', f'{total_biomass:.1f}')

            ff_fish_schools = [int(snap['fish']['num_schools']) for snap in ff_history]
            ff_biomass = [float(snap['fish']['energies'].sum()) if snap['fish']['num_schools'] > 0 else 0.0 for snap in ff_history]

            st.subheader('Fish School Count Over Time')
            fish_school_chart = {
                'step': [snap['interactive_step'] for snap in ff_history],
                'fish_schools': ff_fish_schools,
            }
            st.line_chart(fish_school_chart, x='step')

            st.subheader('Fish Biomass Over Time')
            fish_biomass_chart = {
                'step': [snap['interactive_step'] for snap in ff_history],
                'fish_biomass': ff_biomass,
            }
            st.line_chart(fish_biomass_chart, x='step')

    # Display all three scenarios
    display_scenario('Normal', tab_normal, ff_sims['normal'], ff_histories['normal'], show_boat_highlight=True)
    display_scenario('No Boats', tab_no_boats, ff_sims['no_boats'], ff_histories['no_boats'], show_boat_highlight=False)
    display_scenario('No MPA', tab_no_mpa, ff_sims['no_mpa'], ff_histories['no_mpa'], show_boat_highlight=True)

    st.stop()

if capture_end_day < capture_start_day:
    st.warning('GIF End Day is less than Start Day. End day will be treated as Start Day.')

if mpa_end_day < mpa_start_day:
    st.warning('MPA End Day is less than Start Day. End day will be treated as Start Day.')

if sum(default_boats) <= 0:
    st.error('Please assign at least one boat across ports.')

if run_btn and sum(default_boats) > 0:
    with st.spinner('Running simulation and generating GIF...'):
        t0 = time.time()
        progress_text = st.empty()
        progress_bar = st.progress(0)
        last_pct = {'value': -1}

        def on_progress(done_steps, total_steps):
            pct = int((done_steps / max(1, total_steps)) * 100)
            if pct != last_pct['value']:
                last_pct['value'] = pct
                progress_bar.progress(min(100, pct))
                progress_text.info(f'Simulation progress: {pct}% ({done_steps}/{total_steps} steps)')

        sim = HeuristicSimulation(
            env_width=env_width,
            env_height=env_height,
            hours_per_tick=1,
            initial_fish=initial_fish,
            num_ports=num_ports,
            boats_per_port=default_boats,
            port_locations=custom_port_locations,
            simulation_days=simulation_days,
            mpa_enabled=mpa_enabled,
            seed=int(random_seed),
        )

        result = sim.run(
            capture_start_day=float(capture_start_day),
            capture_end_day=float(capture_end_day),
            target_frames=int(target_frames),
            fps=int(gif_fps),
            media_format=media_format,
            mpa_start_day=float(mpa_start_day),
            mpa_end_day=float(mpa_end_day),
            highlight_boat=None if highlight_boat == -1 else int(highlight_boat),
            focus_port=None if focus_port == -1 else int(focus_port),
            metrics_interval_steps=int(metrics_interval_steps),
            output_dir='outputs',
            run_name=f'ui_run_{int(time.time())}',
            progress_callback=on_progress,
        )
        elapsed = time.time() - t0

    progress_bar.progress(100)
    progress_text.success('Simulation complete: 100%')
    st.session_state['last_result'] = {
        'result': result,
        'elapsed': elapsed,
        'boat_home_port_index': sim.boat_home_port_index,
        'chart_max_points': int(chart_max_points),
    }

if st.session_state['last_result'] is not None:
    cached = st.session_state['last_result']
    result = cached['result']
    elapsed = cached['elapsed']
    boat_home_port_index = cached['boat_home_port_index']
    chart_max_points = cached['chart_max_points']

    summary = result['summary']
    metrics = result['metrics']
    port_report = result['port_report']

    media_label = 'Video' if result.get('media_type') == 'video/mp4' else 'GIF'
    st.success(f"Simulation complete in {elapsed:.1f}s. {media_label} saved at: {result['media_path']}")
    if result.get('fallback_used'):
        st.warning(result.get('fallback_message', 'Media fallback used.'))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Boats', summary['total_boats'])
    c2.metric('Total Catch (tons)', f"{summary['total_catch']:.1f}")
    c3.metric('Final Fish Schools', summary['final_fish_schools'])
    c4.metric('Final Plankton Total', f"{summary['final_plankton_total']:.1f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric('Boats Out of Fuel (final)', summary['boats_out_of_fuel_final'])
    c6.metric('Captured Frames', summary['captured_frames'])
    c7.metric('GIF FPS', summary['fps'])
    c8.metric('Frame Stride', summary['frame_stride'])

    c9, c10 = st.columns(2)
    last_total_biomass = float(metrics['fish_biomass_total'][-1]) if len(metrics['fish_biomass_total']) > 0 else 0.0
    last_total_energy = float(metrics['total_energy'][-1]) if len(metrics['total_energy']) > 0 else 0.0
    last_captured_biomass = float(metrics['captured_biomass'][-1]) if len(metrics['captured_biomass']) > 0 else 0.0
    c9.metric('Total Biomass', f'{last_total_biomass:.1f}')
    c10.metric('Total Energy', f'{last_total_energy:.1f}')
    st.metric('Captured Biomass', f'{last_captured_biomass:.1f}t')

    media_col, replay_col = st.columns([6, 1])
    with media_col:
        st.subheader('Simulation Playback')
    with replay_col:
        if st.button('Replay', help='Reload media from start'):
            st.session_state['media_nonce'] = st.session_state.get('media_nonce', 0) + 1
            st.rerun()

    with open(result['media_path'], 'rb') as media_file:
        media_bytes = media_file.read()

    if result.get('media_type') == 'video/mp4':
        st.video(media_bytes)
    else:
        st.image(media_bytes, caption='Fleet playback for selected day range')

    st.subheader('Fish School Metrics')
    fish_step, fish_schools = _downsample_xy(metrics['step'], metrics['fish_schools'], chart_max_points)
    fish_delta_step, fish_delta = _downsample_xy(metrics['step'], metrics['fish_school_delta_prev'], chart_max_points)
    fish_change_step, fish_change = _downsample_xy(metrics['step'], metrics['fish_school_change_from_start'], chart_max_points)
    fish_biomass_step, fish_biomass_total = _downsample_xy(metrics['step'], metrics['fish_biomass_total'], chart_max_points)
    fish_biomass_change_step, fish_biomass_change = _downsample_xy(metrics['step'], metrics['fish_biomass_change_from_start'], chart_max_points)
    fish_energy_step, fish_avg_energy = _downsample_xy(metrics['step'], metrics['avg_fish_school_energy'], chart_max_points)
    fish_captured_step, fish_captured_biomass = _downsample_xy(metrics['step'], metrics['captured_biomass'], chart_max_points)
    fish_total_energy_step, fish_total_energy = _downsample_xy(metrics['step'], metrics['total_energy'], chart_max_points)
    fish_chart = {
        'step': fish_step,
        'fish_schools': fish_schools,
        'fish_school_delta_prev': fish_delta,
        'fish_school_change_from_start': fish_change,
        'fish_biomass_total': fish_biomass_total,
        'fish_biomass_change_from_start': fish_biomass_change,
        'avg_fish_school_energy': fish_avg_energy,
        'captured_biomass': fish_captured_biomass,
        'total_energy': fish_total_energy,
    }
    st.line_chart(fish_chart, x='step')

    st.subheader('Core Metrics Over Time')
    core_step, plankton_total = _downsample_xy(metrics['step'], metrics['plankton_total'], chart_max_points)
    core_step2, cumulative_catch = _downsample_xy(metrics['step'], metrics['cumulative_catch'], chart_max_points)
    core_step3, fleet_avg_fuel = _downsample_xy(metrics['step'], metrics['fleet_avg_fuel'], chart_max_points)
    core_step4, boats_out = _downsample_xy(metrics['step'], metrics['boats_out_of_fuel'], chart_max_points)
    core_step5, inside_mpa = _downsample_xy(metrics['step'], metrics['inside_mpa_boats'], chart_max_points)
    line_data = {
        'step': core_step,
        'plankton_total': plankton_total,
        'cumulative_catch': cumulative_catch[:len(core_step)],
        'fleet_avg_fuel': fleet_avg_fuel[:len(core_step)],
        'boats_out_of_fuel': boats_out[:len(core_step)],
        'inside_mpa_boats': inside_mpa[:len(core_step)],
    }
    st.line_chart(line_data, x='step')

    st.subheader('Selected Boat Metrics')
    selected_boat = st.selectbox(
        'Choose Boat',
        options=list(range(summary['total_boats'])),
        format_func=lambda b: f'Boat {b} (Port {boat_home_port_index[b]})',
    )

    boat_step = metrics['step']
    boat_fuel = [row[selected_boat] for row in metrics['boat_fuel_by_step']]
    boat_cargo = [row[selected_boat] for row in metrics['boat_cargo_by_step']]
    boat_step_catch = [row[selected_boat] for row in metrics['boat_step_catch_by_step']]
    boat_cum_catch = [row[selected_boat] for row in metrics['boat_cum_catch_by_step']]

    boat_step_ds, boat_fuel_ds = _downsample_xy(boat_step, boat_fuel, chart_max_points)
    _, boat_cargo_ds = _downsample_xy(boat_step, boat_cargo, chart_max_points)
    _, boat_step_catch_ds = _downsample_xy(boat_step, boat_step_catch, chart_max_points)
    _, boat_cum_catch_ds = _downsample_xy(boat_step, boat_cum_catch, chart_max_points)

    boat_line_data = {
        'step': boat_step_ds,
        'boat_fuel': boat_fuel_ds,
        'boat_cargo': boat_cargo_ds,
        'boat_step_catch': boat_step_catch_ds,
        'boat_cumulative_catch': boat_cum_catch_ds,
    }
    st.line_chart(boat_line_data, x='step')

    st.subheader('Port Performance')
    st.table(port_report)

    if focus_port != -1:
        port_idx = int(focus_port)
        row = [r for r in port_report if r['port_index'] == port_idx]
        if len(row) == 1:
            st.info(
                f"Port {port_idx}: boats={row[0]['boats']}, total_catch={row[0]['total_catch']:.1f}t, "
                f"avg_catch_per_boat={row[0]['avg_catch_per_boat']:.1f}t, out_of_fuel={row[0]['boats_out_of_fuel']}"
            )

    with st.expander('Run Configuration (JSON)'):
        st.code(json.dumps({
            'simulation_days': simulation_days,
            'env_width': env_width,
            'env_height': env_height,
            'initial_fish': initial_fish,
            'num_ports': num_ports,
            'use_custom_ports': use_custom_ports,
            'custom_port_locations': custom_port_locations,
            'boats_per_port': default_boats,
            'focus_port': focus_port,
            'capture_start_day': capture_start_day,
            'capture_end_day': capture_end_day,
            'target_frames': target_frames,
            'gif_fps': gif_fps,
            'media_format': media_format,
            'mpa_start_day': mpa_start_day,
            'mpa_end_day': mpa_end_day,
            'metrics_interval_steps': metrics_interval_steps,
            'chart_max_points': chart_max_points,
            'mpa_enabled': mpa_enabled,
            'seed': int(random_seed),
        }, indent=2), language='json')
else:
    st.info('Set parameters in the sidebar and click Run Simulation.')
