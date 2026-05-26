"""
Fishing Fleet Deployment (Heuristic / Training-Free)

Runs the same fish + environment + MPA simulation pipeline,
but uses deterministic rule-based fleet controllers instead of RL checkpoints.
"""

from deploy_fleet import FleetDeployer


if __name__ == "__main__":
    config = {
        # Environment
        'env_width': 100,
        'env_height': 100,
        'hours_per_tick': 1,
        'initial_fish': 2500,

        # Fleet
        'num_ports': 5,

        # Deployment
        'deployment_steps': 8760,

        # Controller mode
        'controller_mode': 'heuristic',
    }

    print("\n" + "=" * 70)
    print("FISHING FLEET DEPLOYMENT SYSTEM (HEURISTIC / TRAINING-FREE)")
    print("=" * 70)
    print("\n🚀 Preparing to deploy 15 boats (5 groups × 3 instances each)")
    print("📍 Random port locations will be generated")
    print(f"⏱️  Deployment duration: {config['deployment_steps']} hours (1 year / 365 days)")
    print("🧠 Controller mode: heuristic (no training, no checkpoints)")
    print("\n📊 Each GIF will show only the 3 boats for that controller group")
    print("=" * 70)

    try:
        input("\nPress Enter to start deployment...")
    except KeyboardInterrupt:
        print("\nDeployment cancelled.")
        raise SystemExit(0)

    try:
        deployer = FleetDeployer(config)
        deployer.deploy()
        deployer.generate_gifs()
        deployer.print_deployment_summary()

        print("🎉 Heuristic deployment completed successfully!")
        print("Check deployment_model_X_fleet.gif files for visualizations.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during deployment: {e}")
        import traceback
        traceback.print_exc()
