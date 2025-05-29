import jax.numpy as jnp

from chromatix import Field
from chromatix.field import ScalarField, VectorField, cartesian_to_spherical
from chromatix.functional.amplitude_masks import amplitude_change
from chromatix.functional.convenience import optical_fft
from chromatix.functional.phase_masks import phase_change
from chromatix.functional.rays import (
    compute_free_space_abcd,
    compute_plano_convex_spherical_lens_abcd,
    ray_transfer,
)
from chromatix.typing import Array, ScalarLike
from chromatix.utils.czt import zoomed_fft

from ..utils import l2_sq_norm
from ..utils.initializers import (
    hexagonal_microlens_array_amplitude_and_phase,
    microlens_array_amplitude_and_phase,
    rectangular_microlens_array_amplitude_and_phase,
)
from .pupils import circular_pupil

__all__ = [
    "thin_lens",
    "ff_lens",
    "df_lens",
    "microlens_array",
    "hexagonal_microlens_array",
    "rectangular_microlens_array",
    "thick_plano_convex_lens",
    "thick_plano_convex_ff_lens",
    "high_na_ff_lens",
]


def thin_lens(
    field: Field, f: ScalarLike, n: ScalarLike, NA: ScalarLike | None = None
) -> Field:
    """
    Applies a thin lens placed directly after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` directly after the lens.
    """
    L = jnp.sqrt(field.spectrum * f / n)
    phase = -jnp.pi * l2_sq_norm(field.grid) / L**2

    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    return field * jnp.exp(1j * phase)


def ff_lens(
    field: Field,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike | None = None,
    inverse: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` to and after the lens.
    """
    # Pupil
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    if inverse:
        # if inverse, propagate over negative distance
        f = -f
    return optical_fft(field, f, n)


def high_na_ff_lens(
    field: Field,
    f: float,
    n: float,
    NA: float,
    output_shape: int | tuple[int, int] | None = None,
    output_dx: float | tuple[float, float] | None = None,
    defocus=0,
    s_z_correction: bool = True,
    apodization: bool = True,
    gibson_lanni: bool = False,
    t_s=0.1e3,
    t_i0=100e3,
    t_g=150e3,
    t_g0=150e3,
    n_s=1.3,
    n_i=1.5,
    n_i0=1.5,
    n_g=1.5,
    n_g0=1.5,
) -> Field:
    """
    Applies a high NA lens placed a distance ``f`` after the incoming ``Field``.

    !!!warning
        This function assumes that the incoming ``Field`` contains only a single
        wavelength.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        output_shape: The shape of the camera (in pixels). If not provided, the
            output shape will be the same as the shape of the incoming field.
        output_dx: The pixel pitch of the camera (in units of distance). If not
            provided, the output spacing will be the same as the spacing of the
            incoming field.

    Returns:
        The ``Field`` propagated a distance ``f`` to and after the lens.
    """
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    if field.shape[-1] == 1:
        # Scalar
        spherical_u = field.u
        create = ScalarField.create
    else:
        # Vectorial
        spherical_u = cartesian_to_spherical(field, n, NA, f)
        create = VectorField.create

    if output_dx is None:
        output_dx = field.spectrum.squeeze() / NA / 2
    if output_dx.ndim == 0:
        output_dx = jnp.stack([output_dx, output_dx])

    if output_shape is None:
        output_shape = field.spatial_shape
    if isinstance(output_shape, int):
        output_shape = jnp.stack([output_shape, output_shape])

    # NOTE: This only works for single wavelength so far?
    fov_out = tuple(o * dx for o, dx in zip(output_shape, output_dx))
    zoom_factor = tuple(
        2 * NA * fov / ((min(field.shape[1:3]) - 1) * field.spectrum.squeeze())
        for fov in fov_out
    )

    sin_theta2 = jnp.clip((field.grid[0] ** 2 + field.grid[1] ** 2) / f**2, 0, 1)
    theta = jnp.arcsin(jnp.sqrt(sin_theta2))
    if s_z_correction:
        spherical_u /= jnp.cos(theta)
    if apodization:
        spherical_u *= jnp.sqrt(jnp.clip(jnp.cos(theta), 0, None))
    if gibson_lanni:
        clamp_value = min(n_s / n_i, n_g / n_i) ** 2
        sin_theta2 = jnp.clip(sin_theta2, 0, clamp_value)
        t_i = n_i * (t_g0 / n_g0 + t_i0 / n_i0 - t_g / n_g - t_s / n_s)
        phase = (
            2
            * jnp.pi
            / field.spectrum.squeeze()
            * (
                t_s * jnp.sqrt(n_s**2 - n_i**2 * sin_theta2)
                + t_i * jnp.sqrt(n_i**2 - n_i**2 * sin_theta2)
                - t_i0 * jnp.sqrt(n_i0**2 - n_i**2 * sin_theta2)
                + t_g * jnp.sqrt(n_g**2 - n_i**2 * sin_theta2)
                - t_g0 * jnp.sqrt(n_g0**2 - n_i**2 * sin_theta2)
            )
        )
        spherical_u *= jnp.exp(1j * phase)

    # Add defocus
    k = 2 * jnp.pi * n / field.spectrum.squeeze()
    spherical_u *= jnp.exp(1j * k * defocus * jnp.cos(theta))
    u = zoomed_fft(
        x=spherical_u,
        k_start=tuple(-z * jnp.pi for z in zoom_factor),
        k_end=tuple(z * jnp.pi for z in zoom_factor),
        output_shape=output_shape,
        include_end=True,
        axes=field.spatial_dims,
        fftshift_input=True,
    )

    # Normalize the output
    L_sq = field.spectrum * f / n
    norm_factor = jnp.prod(field.dx, axis=0, keepdims=False) / L_sq
    u *= norm_factor

    return create(output_dx[:, None], field._spectrum, field._spectral_density, u)


def df_lens(
    field: Field,
    d: ScalarLike,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike | None = None,
    inverse: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``d`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        d: Distance from the incoming ``Field`` to the lens.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    if inverse:
        # if inverse, propagate over negative distance
        f = -d
        d = -f
    field = optical_fft(field, f, n)

    # Phase factor due to distance d from lens
    L = jnp.sqrt(jnp.complex64(field.spectrum * f / n))  # Lengthscale L
    phase = jnp.pi * (1 - d / f) * l2_sq_norm(field.grid) / jnp.abs(L) ** 2
    return field * jnp.exp(1j * phase)


def microlens_array(
    field: Field,
    n: ScalarLike,
    fs: Array,
    centers: Array,
    radii: Array,
    block_between: bool = False,
) -> Field:
    amplitude, phase = microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n,
        fs,
        centers,
        radii,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


def hexagonal_microlens_array(
    field: Field,
    n: ScalarLike,
    f: Array,
    num_lenses_per_side: ScalarLike,
    radius: Array,
    separation: ScalarLike,
    block_between: bool = False,
) -> Field:
    amplitude, phase = hexagonal_microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n,
        f,
        num_lenses_per_side,
        radius,
        separation,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


def rectangular_microlens_array(
    field: Field,
    n: ScalarLike,
    f: Array,
    num_lenses_height: ScalarLike,
    num_lenses_width: ScalarLike,
    radius: Array,
    separation: ScalarLike,
    block_between: bool = False,
) -> Field:
    amplitude, phase = rectangular_microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n,
        f,
        num_lenses_height,
        num_lenses_width,
        radius,
        separation,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


def thick_plano_convex_lens(
    field: Field,
    f: ScalarLike,
    R: ScalarLike,
    center_thickness: ScalarLike,
    n_lens: ScalarLike,
    n_medium: ScalarLike = 1.0,
    NA: ScalarLike | None = None,
    inverse: bool = False,
    magnification: ScalarLike = 1.0,
) -> Field:
    if NA is not None:
        D = 2 * f * NA / n_medium  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    ABCD = compute_plano_convex_spherical_lens_abcd(
        f, R, center_thickness, n_lens, n_medium, inverse
    )
    field = ray_transfer(field, ABCD, n_medium, magnification=magnification)
    return field


def thick_plano_convex_ff_lens(
    field: Field,
    f: ScalarLike,
    R: ScalarLike,
    center_thickness: ScalarLike,
    n_lens: ScalarLike,
    n_medium: ScalarLike = 1.0,
    NA: ScalarLike | None = None,
    inverse: bool = False,
    magnification: ScalarLike = 1.0,
) -> Field:
    if NA is not None:
        D = 2 * f * NA / n_medium  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    _lens = compute_plano_convex_spherical_lens_abcd(
        f, R, center_thickness, n_lens, n_medium, inverse
    )
    _free_space = compute_free_space_abcd(f)
    ABCD = _free_space @ _lens @ _free_space
    field = ray_transfer(field, ABCD, n_medium, magnification=magnification)
    return field
