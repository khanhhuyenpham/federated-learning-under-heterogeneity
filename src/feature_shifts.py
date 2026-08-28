from torchvision.transforms import functional as TF
import numpy as np

class FixedRotation:
    """Rotate every input by a fixed angle in degrees."""

    def __init__(self, angle: float):
        self.angle = float(angle)

    def __call__(self, features):
        return TF.rotate(img=features, angle=self.angle)

def build_client_rotation_transforms(
        num_clients: int,
        max_abs_angle: float,
) -> dict[int, FixedRotation]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    if max_abs_angle < 0:
        raise ValueError("max_abs_angle must be nonnegative")

    if num_clients == 1:
        client_angles = np.array([0.0])
    else:
        client_angles = np.linspace(-max_abs_angle, max_abs_angle, num_clients)
    return {
        client_id : FixedRotation(client_angles[client_id])
        for client_id in range(num_clients)
    }
