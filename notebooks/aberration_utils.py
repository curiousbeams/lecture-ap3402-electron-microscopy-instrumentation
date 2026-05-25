import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import ipywidgets as widgets


def electron_wavelength(energy):
    m = 9.109383e-31  # mass in SI (Kg)
    e = 1.602177e-19  # elementary charge in SI (C)
    c = 299792458  # speed of light in SI (m/s)
    h = 6.62607e-34  # planch constant in SI (Kg m^2 / s)

    lam = h / np.sqrt(2 * m * e * energy) / np.sqrt(1 + e * energy / 2 / m / c**2)
    return lam * 1e10  # convert from m to Angstroms


# aperture functions
def hard_aperture_function(alpha, semiangle_cutoff):
    semiangle_rad = semiangle_cutoff * 1e-3
    aperture = (alpha < semiangle_rad).astype(np.float32)
    return aperture


def soft_aperture_function(alpha, phi, semiangle_cutoff, angular_sampling):
    semiangle_rad = semiangle_cutoff * 1e-3
    denominator = np.sqrt(
        (np.cos(phi) * angular_sampling[0] * 1e-3) ** 2
        + (np.sin(phi) * angular_sampling[1] * 1e-3) ** 2
    )
    aperture = np.clip((semiangle_rad - alpha) / denominator + 0.5, 0, 1)
    return aperture


def aperture_function(alpha, phi, semiangle_cutoff, angular_sampling, soft_edges=True):
    if soft_edges:
        return soft_aperture_function(alpha, phi, semiangle_cutoff, angular_sampling)
    else:
        return hard_aperture_function(
            alpha,
            semiangle_cutoff,
        )


# aberration functions
def aberration_surface(alpha, phi, wavelength, aberration_coefs):
    prefactor = 2 * np.pi / wavelength
    chi = np.zeros_like(alpha)

    alpha2 = alpha**2
    alpha3 = alpha * alpha2
    alpha4 = alpha2 * alpha2
    alpha5 = alpha2 * alpha3
    alpha6 = alpha3 * alpha3

    coefs = aberration_coefs.copy()

    def get(name, default=0.0):
        return coefs.get(name, default)

    if any(k in coefs for k in ("C10", "C12", "phi12")):
        chi = chi + 0.5 * alpha2 * (
            get("C10") + get("C12") * np.cos(2 * (phi - get("phi12")))
        )

    if any(k in coefs for k in ("C21", "phi21", "C23", "phi23")):
        chi = chi + (1 / 3) * alpha3 * (
            get("C21") * np.cos(phi - get("phi21"))
            + get("C23") * np.cos(3 * (phi - get("phi23")))
        )

    if any(k in coefs for k in ("C30", "C32", "phi32", "C34", "phi34")):
        chi = chi + (1 / 4) * alpha4 * (
            get("C30")
            + get("C32") * np.cos(2 * (phi - get("phi32")))
            + get("C34") * np.cos(4 * (phi - get("phi34")))
        )

    if any(k in coefs for k in ("C41", "phi41", "C43", "phi43", "C45", "phi45")):
        chi = chi + (1 / 5) * alpha5 * (
            get("C41") * np.cos(phi - get("phi41"))
            + get("C43") * np.cos(3 * (phi - get("phi43")))
            + get("C45") * np.cos(5 * (phi - get("phi45")))
        )

    if any(k in coefs for k in ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")):
        chi = chi + (1 / 6) * alpha6 * (
            get("C50")
            + get("C52") * np.cos(2 * (phi - get("phi52")))
            + get("C54") * np.cos(4 * (phi - get("phi54")))
            + get("C56") * np.cos(6 * (phi - get("phi56")))
        )

    return chi * prefactor


# spatial frequency utils
def spatial_frequencies(
    gpts,
    sampling,
):
    kx = np.fft.fftfreq(gpts[0], sampling[0]).astype(np.float32)
    ky = np.fft.fftfreq(gpts[1], sampling[1]).astype(np.float32)
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")
    return kxa, kya


def polar_coordinates(
    kxa,
    kya,
):
    k = np.sqrt(kxa**2 + kya**2)
    phi = np.arctan2(kya, kxa)
    return k, phi


def make_checkerboard(gpts, n_blocks=16):
    block = gpts[0] // n_blocks
    row_idx = np.arange(gpts[0])[:, None] // block
    col_idx = np.arange(gpts[1]) // block
    cb = ((row_idx + col_idx) % 2 == 0).astype(float)
    cb = np.roll(cb, block // 2, axis=(0, 1))
    return cb


def quiver_slice(step):
    half = step // 2
    return (slice(half, None, step), slice(half, None, step))


def compute_warp(checkerboard, u_row, u_col):
    """
    Pull-warp the checkerboard by the displacement field.

    """
    row, col = np.indices(checkerboard.shape)

    aperture = map_coordinates(
        np.ones_like(checkerboard, dtype=float),
        [row - u_row, col - u_col],
        order=0,
    ).reshape(checkerboard.shape)

    warped = map_coordinates(
        checkerboard,
        [row - u_row, col - u_col],
        order=0,
        mode="wrap",
    ).reshape(checkerboard.shape)

    return warped, aperture


class AberrationsWidget:
    """
    Interactive aberration visualization.

    Panels
    ------
    0  sin(chi)              — phase rings
    1  quiver plot           — ray displacement field
    2  warped checkerboard   — geometric image distortion
    """

    _SLIDER_SPECS = [
        ("C10", r"$C_{10}$ (Å)", -10, 10, 0.1, 0.0),
        ("C30", r"$C_{30}$ (Å)", -500, 500, 1.0, 0.0),
        ("C12", r"$C_{12}$ (Å)", 0, 10, 0.1, 0.0),
        ("phi12", r"$\phi_{12}$ (rad)", -np.pi / 2, np.pi / 2, np.pi / 48, 0.0),
        ("C21", r"$C_{21}$ (Å)", 0, 100, 1.0, 0.0),
        ("phi21", r"$\phi_{21}$ (rad)", -np.pi, np.pi, np.pi / 48, 0.0),
        ("C23", r"$C_{23$ (Å)", 0, 100, 1.0, 0.0),
        ("phi23", r"$\phi_{23}$ (rad)", -np.pi / 3, np.pi / 3, np.pi / 48, 0.0),
    ]

    def __init__(
        self,
        energy=300e3,
        gpts=(512, 512),
        sampling=(0.1, 0.1),
        n_blocks=16,
        quiver_step=32,
    ):
        # dataclass attributes
        self.energy = energy
        self.gpts = gpts
        self.sampling = sampling
        self.n_blocks = n_blocks
        self.quiver_step = quiver_step

        # computed attributes
        self.wavelength = electron_wavelength(energy)
        kxa, kya = spatial_frequencies(gpts, sampling)
        self.kxa_c = np.fft.fftshift(kxa)
        self.kya_c = np.fft.fftshift(kya)
        self.reciprocal_sampling = tuple(1 / s / g for s, g in zip(sampling, gpts))
        self.angular_sampling = tuple(
            1 / s / g * self.wavelength * 1e3 for s, g in zip(sampling, gpts)
        )

        self.checkerboard = make_checkerboard(gpts, n_blocks)
        self._build_widgets()
        self._build_figure()

    def _build_widgets(self):
        self.sliders = {}
        for name, label, lo, hi, step, val in self._SLIDER_SPECS:
            self.sliders[name] = widgets.FloatSlider(
                value=val,
                min=lo,
                max=hi,
                step=step,
                description=label,
                style={"description_width": "100px"},
                layout=widgets.Layout(width="310px"),
                continuous_update=True,
            )
            self.sliders[name].observe(self._on_change, names="value")

        row1 = widgets.HBox([self.sliders[k] for k in ("C10", "C30")])
        row2 = widgets.HBox([self.sliders[k] for k in ("C12", "phi12")])
        row3 = widgets.HBox([self.sliders[k] for k in ("C21", "phi21")])
        self.controls = widgets.VBox([row1, row2, row3])

    def _build_figure(self):
        aberration_coefs = self._current_coefs()
        k, phi = polar_coordinates(self.kxa_c, self.kya_c)
        alpha = k * self.wavelength
        chi_c = aberration_surface(alpha, phi, self.wavelength, aberration_coefs)

        dchi_drow, dchi_dcol = np.gradient(
            chi_c,
            self.reciprocal_sampling[0],
            self.reciprocal_sampling[1],
            edge_order=2,
        )
        u_row = dchi_drow / self.reciprocal_sampling[0]
        u_col = dchi_dcol / self.reciprocal_sampling[1]

        warped, aperture = compute_warp(self.checkerboard, u_row, u_col)
        sl = quiver_slice(self.quiver_step)

        width = 620
        aspect_ratio = 0.4
        height = int(width * aspect_ratio)
        dpi = 72
        with plt.ioff():
            fig, axs = plt.subplots(1, 3, figsize=(width / dpi, height / dpi), dpi=dpi)
        self.fig = fig
        self.axs = axs

        # --- panel 0: sin(chi) ---
        self._im0 = axs[0].imshow(
            np.sin(chi_c),
            cmap="twilight",
            vmin=-1,
            vmax=1,
        )
        axs[0].set_title(r"$\sin(\chi)$")

        # --- panel 1: quiver ---
        qkx = self.kya_c[sl] * self.wavelength * 1e3
        qky = -self.kxa_c[sl] * self.wavelength * 1e3
        qdx = u_col[sl]
        qdy = -u_row[sl]

        self._quiver = axs[1].quiver(
            qkx, qky, qdx, qdy, color="white", angles="xy", scale_units="xy", scale=10
        )
        axs[1].set_title(r"$\nabla \chi$ ray displacement")

        # --- panel 2: warped checkerboard ---
        self._im2 = axs[2].imshow(
            warped,
            cmap="gray",
            alpha=None,
        )
        self._im2.set_alpha(aperture * 0.5 + 0.5)
        axs[2].set_title("warped checkerboard")

        for ax in axs:
            ax.set(xticks=[], yticks=[], aspect="equal")
            ax.patch.set_alpha(0)

        fig.patch.set_alpha(0)
        fig.tight_layout()
        fig.canvas.resizable = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.layout.width = f"{width}px"
        fig.canvas.toolbar_position = "bottom"
        return None

    def _on_change(self, change):
        aberration_coefs = self._current_coefs()
        k, phi = polar_coordinates(self.kxa_c, self.kya_c)
        alpha = k * self.wavelength
        chi_c = aberration_surface(alpha, phi, self.wavelength, aberration_coefs)

        dchi_drow, dchi_dcol = np.gradient(
            chi_c,
            self.reciprocal_sampling[0],
            self.reciprocal_sampling[1],
            edge_order=2,
        )
        u_row = dchi_drow / self.reciprocal_sampling[0]
        u_col = dchi_dcol / self.reciprocal_sampling[1]

        warped, aperture = compute_warp(self.checkerboard, u_row, u_col)
        sl = quiver_slice(self.quiver_step)

        # panel 0
        self._im0.set_data(np.sin(chi_c))

        # panel 1
        self._quiver.set_UVC(
            u_col[sl],
            -u_row[sl],
        )

        # panel 2
        self._im2.set_data(warped)
        self._im2.set_alpha(aperture * 0.5 + 0.5)
        self.fig.canvas.draw_idle()
        return None

    def _current_coefs(self):
        return {k: v.value for k, v in self.sliders.items() if v.value != 0.0}

    def display(self):
        return widgets.VBox(
            [self.controls, self.fig.canvas],
            layout=widgets.Layout(align_items="center"),
        )
