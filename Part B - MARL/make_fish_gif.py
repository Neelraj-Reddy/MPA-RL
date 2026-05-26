"""
Fish GIF Generator - Similar style to make_gif.py, but fish-only.
Generates an animated GIF of fish schools moving over temperature field.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image
import imageio

from environment import OceanEnvironment
from fish_ecosystem import FishPopulation


class FishGIFMaker:
    def __init__(self, num_steps=500, fps=20):
        self.num_steps = num_steps
        self.fps = fps

        # Setup
        self.env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
        self.fish = FishPopulation(initial_population=2500, env_width=100, env_height=100)

        # Colormap for temperature
        colors = ['#000033', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        self.cmap = LinearSegmentedColormap.from_list('ocean', colors, N=100)

        self.frames = []

        print(f"\n🐟 Fish GIF Generator ready: {num_steps} steps at {fps} FPS")
        print(f"   Expected duration: {num_steps/fps:.1f} seconds")
        print(f"   File size estimate: {(num_steps/fps) * 1.2:.1f} MB\n")

    def simulate_and_capture(self):
        """Run simulation and capture matplotlib frames as PIL images"""
        print("📹 Running simulation and capturing frames...")

        for step in range(self.num_steps):
            # Step environment and fish
            self.fish.step(self.env)
            self.env.step()

            # Create frame occasionally to limit size
            if step % max(1, self.num_steps // 100) == 0 or step < 5:
                frame = self._create_frame(step)
                self.frames.append(frame)

            if (step + 1) % 50 == 0:
                print(f"  Progress: {step + 1}/{self.num_steps} | Fish: {self.fish.num_schools}")

        print(f"✅ Captured {len(self.frames)} frames")

    def _create_frame(self, step):
        """Create a single PIL Image frame"""
        fig = plt.figure(figsize=(12, 8), dpi=80)
        ax = fig.add_subplot(111)

        # Temperature background
        ax.imshow(
            self.env.temperature_grid.T,
            origin='lower',
            cmap=self.cmap,
            extent=[0, 100, 0, 100],
            alpha=0.6,
            vmin=5,
            vmax=25
        )

        # Fish
        if self.fish.num_schools > 0:
            sizes = np.clip(self.fish.energies / 10, 2, 60)
            xs = np.clip(self.fish.positions[:, 0], 0, self.env.width - 1.0)
            ys = np.clip(self.fish.positions[:, 1], 0, self.env.height - 1.0)
            ax.scatter(xs, ys, s=sizes, c='cyan', alpha=0.7,
                       edgecolors='blue', linewidths=0.5)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_title(
            f'Fish Ecosystem - Hour {step}\nSchools: {self.fish.num_schools}',
            fontsize=12,
            fontweight='bold'
        )

        # Convert to PIL Image using buffer approach
        fig.canvas.draw()
        rgba_buf = fig.canvas.buffer_rgba()
        (w, h) = fig.canvas.get_width_height()

        image = Image.frombytes('RGBA', (w, h), rgba_buf)
        image = image.convert('RGB')
        plt.close(fig)

        return image

    def save_gif(self, filename='fish_movement.gif'):
        """Save frames as GIF using imageio"""
        print(f"\n💾 Saving GIF: {filename}")
        print(f"   Frames: {len(self.frames)}")
        print(f"   FPS: {self.fps}")

        frames_array = [np.array(img) for img in self.frames]
        imageio.mimsave(filename, frames_array, fps=self.fps, loop=0)

        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"✅ Saved: {filename} ({file_size_mb:.1f} MB)\n")

        print("=" * 70)
        print("FISH GIF SUMMARY")
        print("=" * 70)
        print(f"Duration: {len(self.frames) / self.fps:.1f} seconds")
        print(f"Resolution: 960x640 pixels")
        print(f"Final Fish Schools: {self.fish.num_schools}")
        print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("FISH MOVEMENT GIF GENERATOR")
    print("=" * 70)

    generator = FishGIFMaker(num_steps=26280, fps=20)
    generator.simulate_and_capture()
    generator.save_gif('fish_movement.gif')

    print("\n🎉 FISH GIF COMPLETE!")
    print("   Open 'fish_movement.gif' to watch the ecosystem evolve.\n")
