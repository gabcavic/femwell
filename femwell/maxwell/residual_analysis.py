"""A posteriori error estimation and residual analysis for finite element solutions."""

from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from skfem import (
    Basis,
    BilinearForm,
    CellBasis,
    ElementDG,
    ElementTriP0,
    ElementTriP1,
    Functional,
    InteriorFacetBasis,
    LinearForm,
)
from skfem.helpers import curl, dot, grad, inner

from .waveguide import Mode


def compute_residual(mode: Mode, mu_r: float = 1, radius: float = np.inf) -> Tuple[NDArray, Basis]:
    """Compute the residual of the eigenvalue problem for a posteriori error estimation.

    The residual is defined as r = (A - λB)u where:
    - A, B are the bilinear forms from the eigenvalue problem
    - λ is the eigenvalue (k^2)
    - u is the eigenvector (electric field E)

    Args:
        mode: The Mode object containing the solution
        mu_r: Relative permeability (default: 1)
        radius: Radius for cylindrical coordinates (default: np.inf for Cartesian)

    Returns:
        Tuple[NDArray, Basis]: The residual vector and the basis
    """
    k0 = mode.k0
    lambda_eigenvalue = mode.k**2

    # Recreate the bilinear forms used in compute_modes
    @BilinearForm(dtype=mode.epsilon_r.dtype)
    def aform_residual(e_t, e_z, v_t, v_z, w):
        epsilon = w.epsilon * (1 + w.x[0] / radius) ** 2
        return (
            1 / mu_r * curl(e_t) * curl(v_t) / k0**2
            - epsilon * dot(e_t, v_t)
            + 1 / mu_r * dot(grad(e_z), v_t)
            + epsilon * inner(e_t, grad(v_z))
            - epsilon * e_z * v_z * k0**2
        )

    @BilinearForm(dtype=mode.epsilon_r.dtype)
    def bform_residual(e_t, e_z, v_t, v_z, w):
        return -1 / mu_r * dot(e_t, v_t) / k0**2

    # Assemble the matrices
    A = aform_residual.assemble(
        mode.basis, epsilon=mode.basis_epsilon_r.interpolate(mode.epsilon_r)
    )
    B = bform_residual.assemble(
        mode.basis, epsilon=mode.basis_epsilon_r.interpolate(mode.epsilon_r)
    )

    # Compute residual r = (A - λB)u
    residual_vector = A @ mode.E - lambda_eigenvalue * (B @ mode.E)

    return residual_vector, mode.basis


def compute_element_residual_indicators(
    mode: Mode, mu_r: float = 1, radius: float = np.inf
) -> Tuple[NDArray, CellBasis]:
    """Compute element-wise residual indicators for a posteriori error estimation.

    This computes element-wise residual norms that can be used as error indicators
    for adaptive mesh refinement.

    Args:
        mode: The Mode object containing the solution
        mu_r: Relative permeability (default: 1)
        radius: Radius for cylindrical coordinates (default: np.inf for Cartesian)

    Returns:
        Tuple[NDArray, CellBasis]: Element-wise error indicators and associated basis
    """
    # Compute the residual vector
    residual_vector, _ = compute_residual(mode, mu_r, radius)

    # Project residual to element-wise constant basis for visualization
    element_indicator_basis = mode.basis.with_element(ElementTriP0())

    @Functional
    def element_indicator(w):
        """Compute element-wise residual norm"""
        # Get element-wise residual contribution
        return (
            np.abs(w.residual[0][0]) ** 2
            + np.abs(w.residual[0][1]) ** 2
            + np.abs(w.residual[1]) ** 2
        )

    # Interpolate residual on elements
    residual_interp = mode.basis.interpolate(residual_vector)
    element_indicators = element_indicator.elemental(
        element_indicator_basis, residual=residual_interp
    )

    return element_indicators, element_indicator_basis


def compute_edge_jump_indicators(mode: Mode) -> NDArray:
    """Compute edge jump indicators for a posteriori error estimation.

    This computes the jumps across element interfaces, which is a key component
    of residual-based error estimators.

    Args:
        mode: The Mode object containing the solution

    Returns:
        NDArray: Element-wise edge jump indicators
    """
    # facet jump computation (similar to existing eval_error_estimator)
    fbasis = [InteriorFacetBasis(mode.basis.mesh, mode.basis.elem, side=i) for i in [0, 1]]
    fbasis_epsilon = [
        InteriorFacetBasis(
            mode.basis.mesh, mode.basis_epsilon_r.elem, side=i, quadrature=fbasis[0].quadrature
        )
        for i in [0, 1]
    ]
    w = {f"u{str(i + 1)}": fbasis[i].interpolate(mode.E) for i in [0, 1]}
    w2 = {f"epsilon{str(i + 1)}": fbasis_epsilon[i].interpolate(mode.epsilon_r) for i in [0, 1]}

    # Normalization factors (can be improved)
    norm_0 = norm_1 = 1

    @Functional
    def edge_jump(w):
        return w.h * (
            np.abs(dot(grad(w["u1"][1]) - grad(w["u2"][1]), w.n)) ** 2 / norm_1**2
            + np.abs(dot(w["u1"][0] * w["epsilon1"][0] - w["u2"][0] * w["epsilon2"][0], w.n)) ** 2
            / norm_0**2
        )

    tmp = np.zeros(mode.basis.mesh.facets.shape[1])
    tmp[fbasis[0].find] = edge_jump.elemental(fbasis[0], **w, **w2)
    eta_E = np.sum(0.5 * tmp[mode.basis.mesh.t2f], axis=0)

    return eta_E


def compute_dual_weighted_residual(
    mode: Mode, quantity_of_interest_functional: Callable, mu_r: float = 1, radius: float = np.inf
) -> float:
    """Compute dual-weighted residual for goal-oriented error estimation.

    Args:
        mode: The Mode object containing the solution
        quantity_of_interest_functional: Functional defining the quantity of interest
        mu_r: Relative permeability (default: 1)
        radius: Radius for cylindrical coordinates (default: np.inf for Cartesian)

    Returns:
        float: Dual-weighted residual estimate
    """
    # Compute primal residual
    residual_vector, _ = compute_residual(mode, mu_r, radius)

    # For proper dual-weighted residual, we would need to solve the dual problem
    # This is a simplified version that assumes the dual solution equals the primal
    # In practice, you would solve: A^T z = quantity_of_interest_functional

    # Simplified estimate using self-duality approximation
    dual_weighted_estimate = np.abs(np.vdot(mode.E, residual_vector))

    return dual_weighted_estimate


def project_residual_on_basis(
    residual_vector: NDArray, source_basis: Basis, target_basis: Basis
) -> NDArray:
    """Project residual vector from one basis to another.

    This is useful for projecting residuals onto different function spaces
    for visualization or further analysis.

    Args:
        residual_vector: The residual vector to project
        source_basis: The basis on which the residual is currently defined
        target_basis: The target basis for projection

    Returns:
        NDArray: The projected residual vector
    """
    # Interpolate residual on quadrature points using source basis
    residual_interp = source_basis.interpolate(residual_vector)

    # Project onto target basis
    projected_residual = target_basis.project(residual_interp)

    return projected_residual


def compute_effectivity_index(
    mode: Mode, exact_error: Optional[float] = None, mu_r: float = 1, radius: float = np.inf
) -> float:
    """Compute the effectivity index of the error estimator.

    The effectivity index is the ratio of estimated error to true error.
    If exact_error is not provided, this returns the estimated error norm.

    Args:
        mode: The Mode object containing the solution
        exact_error: The exact error (if known) for computing effectivity
        mu_r: Relative permeability (default: 1)
        radius: Radius for cylindrical coordinates (default: np.inf for Cartesian)

    Returns:
        float: Effectivity index (if exact_error provided) or estimated error norm
    """
    # Compute residual-based error estimate
    residual_vector, _ = compute_residual(mode, mu_r, radius)
    element_indicators, _ = compute_element_residual_indicators(mode, mu_r, radius)
    edge_indicators = compute_edge_jump_indicators(mode)

    # Combine element and edge contributions
    estimated_error = np.sqrt(np.sum(element_indicators) + np.sum(edge_indicators))

    if exact_error is not None:
        return estimated_error / exact_error
    else:
        return estimated_error


def adaptive_refinement_markers(
    mode: Mode, refinement_fraction: float = 0.3, mu_r: float = 1, radius: float = np.inf
) -> NDArray:
    """Generate element markers for adaptive mesh refinement.

    Args:
        mode: The Mode object containing the solution
        refinement_fraction: Fraction of elements with highest error to mark (default: 0.3)
        mu_r: Relative permeability (default: 1)
        radius: Radius for cylindrical coordinates (default: np.inf for Cartesian)

    Returns:
        NDArray: Boolean array marking elements for refinement
    """
    # Compute error indicators
    element_indicators, _ = compute_element_residual_indicators(mode, mu_r, radius)
    edge_indicators = compute_edge_jump_indicators(mode)

    # Combine indicators (you might want to weight these differently)
    total_indicators = element_indicators + edge_indicators

    # Mark elements with highest errors
    threshold = np.percentile(total_indicators, (1 - refinement_fraction) * 100)
    refinement_markers = total_indicators > threshold

    return refinement_markers
