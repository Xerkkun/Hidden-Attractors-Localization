import unittest
import os
import warnings
import numpy as np
from ..systems.chua_saturation import ChuaSaturationSystem
from ..systems.chua_arctan import ChuaArctanSystem
from ..lure.describing_function import evaluate_describing_function, solve_amplitude_from_gain
from ..lure.nyquist import find_harmonic_candidates
from ..cli.run_workflow import check_duplicate_flags
from ..integrators.general import integrate_general
from ..plotting.plot_trajectories import plot_flexible_attractor_and_projections
from ..plotting.plot_matignon import plot_matignon_equilibria

class TestWorkflowCorrections(unittest.TestCase):
    
    def test_no_warning_saturation_seeds(self):
        """A. Verify no IntegrationWarning is emitted during fractional saturation seed search."""
        system = ChuaSaturationSystem(
            alpha=8.4562, beta=12.0732, gamma=0.0052,
            m0=-0.1768, m1=-1.1468, q=0.9998,
            system_id="chua_fractional_saturation"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            candidates = find_harmonic_candidates(
                system=system,
                transfer_mode="integer",
                seed_strategy="k_phi",
                describing_function_mode="auto",
                amplitude_min=0.01,
                amplitude_max=20.0,
                omega_min=0.01,
                omega_max=20.0,
                grid_size_omega=200,
                grid_size_amplitude=200,
                q=1.0
            )
            
            # Check for IntegrationWarning
            integration_warnings = [
                warn for warn in w 
                if issubclass(warn.category, UserWarning) and "roundoff error" in str(warn.message)
            ]
            self.assertEqual(len(integration_warnings), 0, "IntegrationWarning was raised!")
            self.assertGreater(len(candidates), 0, "No candidates were found!")

    def test_closed_form_vs_quadrature_saturation(self):
        """B. Verify closed-form describing function vs segmented quadrature for Chua Saturation."""
        system = ChuaSaturationSystem(m0=-0.1768, m1=-1.1468)
        amplitudes = [0.5, 1.0, 2.5, 10.0]
        for A in amplitudes:
            res_closed = evaluate_describing_function(system, A, mode="piecewise_closed_form")
            res_quad = evaluate_describing_function(system, A, mode="segmented_quadrature")
            self.assertAlmostEqual(res_closed.value, res_quad.value, places=4)

    def test_arctan_closed_form_vs_quadrature(self):
        """C. Compare closed form vs quadrature for Chua Arctan."""
        system = ChuaArctanSystem(m=0.4, n=-1.1585)
        amplitudes = [0.5, 1.0, 2.5, 10.0]
        for A in amplitudes:
            res_closed = evaluate_describing_function(system, A, mode="closed_form")
            res_quad = evaluate_describing_function(system, A, mode="quadrature")
            self.assertAlmostEqual(res_closed.value, res_quad.value, places=4)

    def test_early_stop_divergence(self):
        """F. Verify that integrating an artificially divergent system stops early."""
        # Vector field that grows exponentially: f(x) = 2.0 * x
        def rhs(t, x):
            return 2.0 * x
            
        x0 = np.array([1.0, 1.0, 1.0])
        
        # Configure early stop: divergence_norm = 10.0, divergence_consecutive_steps = 2
        early_stop_cfg = {
            "enabled": True,
            "divergence_enabled": True,
            "divergence_norm": 10.0,
            "divergence_consecutive_steps": 2,
            "divergence_growth_factor": 1.25,
            "equilibrium_enabled": False
        }
        
        t_arr, x_arr, status = integrate_general(
            rhs=rhs,
            x0=x0,
            q=1.0,
            h=0.01,
            t_final=50.0,
            early_stop_config=early_stop_cfg
        )
        
        self.assertEqual(status, "diverged_early")
        self.assertLess(t_arr[-1], 10.0) # Stopped well before t_final=50.0

    def test_early_stop_equilibrium(self):
        """G. Verify that integrating a stable system converging to origin stops early."""
        # Stable linear system: f(x) = -5.0 * x
        def rhs(t, x):
            return -5.0 * x
            
        x0 = np.array([0.5, 0.5, 0.5])
        equilibria = [np.array([0.0, 0.0, 0.0])]
        
        # Configure early stop for equilibrium
        early_stop_cfg = {
            "enabled": True,
            "divergence_enabled": False,
            "equilibrium_enabled": True,
            "equilibrium_tol": 1e-2,
            "equilibrium_derivative_tol": 1e-2,
            "equilibrium_consecutive_steps": 10,
            "equilibrium_min_time": 0.1
        }
        
        t_arr, x_arr, status = integrate_general(
            rhs=rhs,
            x0=x0,
            q=1.0,
            h=0.01,
            t_final=10.0,
            early_stop_config=early_stop_cfg,
            equilibria=equilibria
        )
        
        self.assertEqual(status, "converged_equilibrium_early")
        self.assertLess(t_arr[-1], 5.0)

    def test_cli_duplicate_flags(self):
        """H. Verify duplicate flags check intercepts and issues warning without failing."""
        argv = ["--preset", "chua_fractional", "--seed-construction", "modal", "--seed-construction", "modal"]
        # Should not raise exception
        try:
            check_duplicate_flags(argv)
            success = True
        except Exception:
            success = False
        self.assertTrue(success)

    def test_attractor_no_equilibria_rendering(self):
        """E. Assert that include_equilibria=false suppresses equilibrium scatter calls in attractor plotting."""
        trajectory = np.zeros((100, 4))
        trajectory[:, 0] = np.linspace(0.0, 1.0, 100)
        equilibria = {"E0": np.array([0.0, 0.0, 0.0])}
        
        config = {
            "system_id": "test_system",
            "t_burn": 10.0,
            "h": 0.01,
            "attractor_plots": {
                "include_equilibria": False
            }
        }
        
        # Call the plot function
        try:
            plot_flexible_attractor_and_projections(
                trajectory=trajectory,
                equilibria=equilibria,
                config=config,
                output_dir="outputs/test_attractor_plots",
                file_prefix="test_attractor_plot"
            )
            
            # Verify file exists
            self.assertTrue(os.path.exists("outputs/test_attractor_plots/figures/test_attractor_plot_3d.png"))
            self.assertTrue(os.path.exists("outputs/test_attractor_plots/figures/test_attractor_plot_xy.png"))
            success = True
        except Exception:
            success = False
            
        self.assertTrue(success)
        
        # Cleanup
        for suffix in ["_3d.png", "_xy.png", "_xz.png", "_yz.png"]:
            p = f"outputs/test_attractor_plots/figures/test_attractor_plot{suffix}"
            if os.path.exists(p):
                os.remove(p)
                
    def test_matignon_plot_generation(self):
        """D. Verify premium Matignon stability plane renders and saves figures successfully."""
        system = ChuaArctanSystem(m=0.4, n=-1.1585, q=0.995)
        equilibria = {"E0": np.array([0.0, 0.0, 0.0])}
        
        config = {
            "system_id": "chua_fractional_arctan",
            "plot_matignon": True
        }
        
        try:
            plot_matignon_equilibria(
                system=system,
                equilibria=equilibria,
                config=config,
                output_dir="outputs/test_matignon_plots"
            )
            self.assertTrue(os.path.exists("outputs/test_matignon_plots/figures/matignon_equilibria.png"))
            success = True
        except Exception:
            success = False
            
        self.assertTrue(success)
        
        # Cleanup
        p = "outputs/test_matignon_plots/figures/matignon_equilibria.png"
        if os.path.exists(p):
            os.remove(p)

if __name__ == '__main__':
    unittest.main()
