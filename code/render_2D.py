import warnings
import math
import fresnel
import IPython
import packaging.version

class render:
    def __init__(self, snapshot, device, w, h):
        self.snapshot = snapshot # snapshot of frames in the simulation
        self.device = device 
        self.w = w # displaying window width
        self.h = h # displaying window height
        self.tracer = fresnel.tracer.Path(device, w, h)
    
    def get_image(self):
        snapshot = self.snapshot
        FRESNEL_MIN_VERSION = packaging.version.parse('0.13.0')
        FRESNEL_MAX_VERSION = packaging.version.parse('0.14.0')
        if (
            'version' not in dir(fresnel)
            or packaging.version.parse(fresnel.version.version) < FRESNEL_MIN_VERSION
            or packaging.version.parse(fresnel.version.version) >= FRESNEL_MAX_VERSION
        ):
            warnings.warn(
                f'Unsupported fresnel version {fresnel.version.version} - expect errors.'
            )
        L = snapshot.configuration.box[0]
        scene = fresnel.Scene(self.device)
        geometry = fresnel.geometry.Sphere(
            scene, N=len(snapshot.particles.position), radius=0.3
        )
        geometry.material = fresnel.material.Material(
            color=fresnel.color.linear([252 / 255, 209 / 255, 1 / 255]), roughness=0.5
        )
        geometry.position[:] = snapshot.particles.position[:]
        geometry.outline_width = 0.04
        fresnel.geometry.Box(scene, [L, L, 0, 0, 0, 0], box_radius=0.02)

        scene.lights = [
            fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8), theta=math.pi),
            fresnel.light.Light(
                direction=(1, 1, 1), color=(1.1, 1.1, 1.1), theta=math.pi / 3
            ),
        ]

        scene.camera = fresnel.camera.Orthographic(
            position=(0, 0, 1*4), look_at=(0, 0, 0), up=(0, 1, 0), height=L * 1.4 + 2
        )
        scene.background_alpha = 1
        scene.background_color = (1, 1, 1)
        return IPython.display.Image(self.tracer.sample(scene, samples=500)._repr_png_())