"""
Fishing Fleet Deployment (MPA OFF)

Same deployment flow as deploy_fleet.py, but Marine Protected Areas are disabled.
"""

import numpy as np
from datetime import timedelta

from deploy_fleet import FleetDeployer


class FleetDeployerNoMPA(FleetDeployer):
    def __init__(self, deployment_config):
        super().__init__(deployment_config)
        self._disable_mpas()

    def _disable_mpas(self):
        self.env.mpa_coverage_target = 0.0
        self.env.mpa_persistence = 0.0
        if hasattr(self.env, 'mpa_grid'):
            self.env.mpa_grid[:] = 0.0

        def no_mpa_update(_fish_population):
            return

        self.env.update_mpas = no_mpa_update

        self.log_file.write("\n[MPA OVERRIDE] MPAs are DISABLED for this deployment run.\n")
        self.log_file.flush()

        print("\n" + "=" * 70)
        print("MPA MODE: OFF")
        print("- mpa_coverage_target set to 0.0")
        print("- update_mpas overridden (no-op)")
        print("=" * 70 + "\n")


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
        'deployment_steps': 8760,  # 1 year (365 days * 24 hours)
    }

    print("\n" + "=" * 70)
    print("FISHING FLEET DEPLOYMENT SYSTEM (MPA OFF)")
    print("=" * 70)
    print(f"\n🚀 Preparing to deploy 15 boats (5 models × 3 instances each)")
    print(f"📍 Random port locations will be generated")
    print(f"⏱️  Deployment duration: {config['deployment_steps']} hours (1 year / 365 days)")
    print("🛑 MPAs are disabled for this run")
    print(f"\n📊 Each GIF will show only the 3 boats for that trained model")
    print("=" * 70)

    try:
        input("\nPress Enter to start deployment...")
    except KeyboardInterrupt:
        print("\nDeployment cancelled.")
        exit(0)

    try:
        deployer = FleetDeployerNoMPA(config)
        deployer.deploy()
        deployer.generate_gifs()
        deployer.print_deployment_summary()

        print("🎉 Deployment completed successfully (MPA OFF)!")
        print("Check deployment_model_X_fleet.gif files for visualizations.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during deployment: {e}")
        import traceback
        traceback.print_exc()
