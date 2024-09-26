import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.sparse
import scipy
import scipy.sparse.linalg

import dolfinx.fem

from global_dofs_manager import GlobalDofsManager
from utils import write_solution

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg

type SparseMatrix = scipy.sparse._csr.csr_matrix

class residual_tracker:

    """Class for tracking the iterative solver residuals and interations.
    """

    def __init__(self):
        self.niter = 0
        self.res = []

    def __call__(self, rk=None):
        self.niter += 1
        self.res.append(rk)


class Assembler:

    """Class for managing the assembling and use of the fetidp operators.
    """

    def __init__(
        self, 
        gbl_dofs_mngr: GlobalDofsManager
    ) -> None: 
        
        """Initializes the class.

        Args:
            gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        """
        
        self.gbl_dofs_mngr = gbl_dofs_mngr
        self.assemble_subdomain()
        self.assemble_primal_Schur()
        self.assemble_dbar()

    def assemble_subdomain(self) -> None:

        """Assembles and stores the subdomain operators.
        """

        subdomain = self.gbl_dofs_mngr.subdomain

        K, f = subdomain.K, subdomain.f

        rem_dofs = subdomain.get_remainder_dofs()
        primal_dofs = subdomain.get_primal_dofs()

        int_dofs, dual_dofs = subdomain.get_prec_dofs()

        Kr = K[rem_dofs, :]
        Krr = Kr[:, rem_dofs]
        self.Krp = Kr[:, primal_dofs]
        Kp = K[primal_dofs, :]
        self.Kpp = Kp[:, primal_dofs]
        self.invKrr = scipy.sparse.linalg.inv(Krr)

        Ki = K[int_dofs, :]
        Kii = Ki[:, int_dofs]
        self.Kid = Ki[:, dual_dofs]
        Kd = K[dual_dofs, :]
        self.Kdd = Kd[:, dual_dofs]
        self.invKii = scipy.sparse.linalg.inv(Kii)

        self.Sdd = self.Kdd - self.Kid.T @ self.invKii @ self.Kid

        self.fr = f[rem_dofs]
        self.fp = f[primal_dofs]

        self.Tdr = subdomain.create_Tdr()

        Urp = self.invKrr @ self.Krp
        self.Spp = self.Kpp - self.Krp.T @ Urp

        self.Aps = []
        self.Bds = []
        self.Brs = []
        self.KrPs = []

        N = self.gbl_dofs_mngr.get_num_subdomains()
        for s_id in range(N):
            Ap = self.gbl_dofs_mngr.create_Ap(s_id)
            Bd = self.gbl_dofs_mngr.create_Bd(s_id)
            self.Aps.append(Ap)
            self.Bds.append(Bd)
            self.Brs.append(Bd @ self.Tdr)
            self.KrPs.append(self.Krp @ Ap.T)

    def assemble_primal_Schur(self) -> None:

        """Assembles and stores the primal Schur operator.
        """

        P = self.gbl_dofs_mngr.get_num_primals()
        act_primal_dofs = self.gbl_dofs_mngr.get_active_primal_dofs()

        SPP = scipy.sparse.csr_matrix((P, P), dtype=self.invKrr.dtype)

        N = self.gbl_dofs_mngr.get_num_subdomains()
        for s_id in range(N):
            Ap = self.Aps[s_id]
            SPP = SPP + Ap @ self.Spp @ Ap.T

        SPP = SPP[act_primal_dofs, :]
        self.SPP = SPP[:, act_primal_dofs]

    def assemble_dbar(self) -> None:

        """Assembles the right hand side of the global dual problem.
        """

        N = self.gbl_dofs_mngr.get_num_subdomains()
        P = self.gbl_dofs_mngr.get_num_primals()
        n_dual = self.gbl_dofs_mngr.get_num_duals()
        act_primal_dofs = self.gbl_dofs_mngr.get_active_primal_dofs()

        y = np.zeros((P,))

        for s_id in range(N):
            Ap = self.Aps[s_id]
            KrP = self.KrPs[s_id]

            y = y + Ap @ self.fp - KrP.T @ self.invKrr @ self.fr

        sol = scipy.sparse.linalg.spsolve(self.SPP, y[act_primal_dofs])
        y = np.zeros((P,))
        y[act_primal_dofs] = sol
        dbar = np.zeros((n_dual,))

        for s_id in range(N):
            KrP = self.KrPs[s_id]
            Br = self.Brs[s_id]

            subdomain_y = self.fr - KrP @ y

            dbar = dbar + Br @ self.invKrr @ subdomain_y

        self.dbar = dbar

    def apply_M(self, x: np.ndarray) -> np.ndarray: 

        """Dirichlet preconditioner.

        Args:
            x (np.ndarray): Vector to which the preconditioner is applied.

        Returns:
            y (np.ndarray): M * x.
        """

        N = self.gbl_dofs_mngr.get_num_subdomains()
        n_dual = self.gbl_dofs_mngr.get_num_duals()

        y = np.zeros((n_dual,))

        for s_id in range(N):
            Bd = self.Bds[s_id]

            y = y + Bd @ self.Sdd @ Bd.T @ x

        return y
    
    def apply_F(self, x: np.ndarray) -> np.ndarray: 

        """Global dual problem left hand side operator.

        Args:
            x (np.ndarray): Vector to which F is applied.

        Returns:
            y (np.ndarray): F * x.
        """

        N = self.gbl_dofs_mngr.get_num_subdomains()
        P = self.gbl_dofs_mngr.get_num_primals()
        n_dual = self.gbl_dofs_mngr.get_num_duals()
        act_primal_dofs = self.gbl_dofs_mngr.get_active_primal_dofs()

        y1 = np.zeros((n_dual,))

        for s_id in range(N):
            Br = self.Brs[s_id]

            y1 = y1 + Br @ self.invKrr @ Br.T @ x

        y2 = np.zeros((P,))

        for s_id in range(N):
            Br = self.Brs[s_id]
            KrP = self.KrPs[s_id]

            y2 = y2 + KrP.T @ self.invKrr @ Br.T @ x

        sol = scipy.sparse.linalg.spsolve(self.SPP, y2[act_primal_dofs])
        y2 = np.zeros((P,))
        y2[act_primal_dofs] = sol

        y = np.zeros((n_dual,))

        for s_id in range(N):
            Br = self.Brs[s_id]
            KrP = self.KrPs[s_id]

            y = y + Br @ self.invKrr @ KrP @ y2

        return y + y1
    
    def reconstruct_uP(
        self,    
        lambda_: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        
        """Reconstructs the global primal solution vector from the multipliers
        lambda_ at the interfaces (the solution of the global dual problem).

        Args:
            lambda_ (npt.NDArray[np.float64]): Dual problem solution vector.

        Returns:
            npt.NDArray[np.float64]: Global primal solution vector.
        """
        
        N = self.gbl_dofs_mngr.get_num_subdomains()
        P = self.gbl_dofs_mngr.get_num_primals()
        act_primal_dofs = self.gbl_dofs_mngr.get_active_primal_dofs()

        y = np.zeros((P,))

        for s_id in range(N):
            Ap = self.Aps[s_id]
            Br = self.Brs[s_id]
            KrP = self.KrPs[s_id]

            y = y + Ap @ self.fp + KrP.T @ self.invKrr @ (Br.T @ lambda_ - self.fr)

        sol = scipy.sparse.linalg.spsolve(self.SPP, y[act_primal_dofs])
        uP = np.zeros((P,))
        uP[act_primal_dofs] = sol

        return uP
    
    def reconstruct_Us(
        self, 
        uP: npt.NDArray[np.float64],
        lambda_: npt.NDArray[np.float64],
    ) -> list[npt.NDArray[np.float64]]:
        
        """Reconstructs the full solution vector of every subdomain, once the
        multipliers lambda_ at the interfaces (the solution of the global dual
        problem), and the global primal solution uP have been computed.

        Args:
            uP (npt.NDArray[np.float64]): Global primal solution vector.
            lambda_ (npt.NDArray[np.float64]): Dual problem solution vector.


        Returns:
            list[npt.NDArray[np.float64]]: Vector of solutions for every subdomain.
        """
        
        N = self.gbl_dofs_mngr.get_num_subdomains()

        subdomain = self.gbl_dofs_mngr.subdomain
        rem_dofs = subdomain.get_remainder_dofs()
        primal_dofs = subdomain.get_primal_dofs()

        us = []

        for s_id in range(N):
            Ap = self.Aps[s_id]
            Br = self.Brs[s_id]
            KrP = self.KrPs[s_id]

            u = np.zeros((rem_dofs.size + primal_dofs.size,))
            u[primal_dofs] = Ap.T @ uP
            u[rem_dofs] = self.invKrr @ (self.fr - KrP @ uP - Br.T @ lambda_)
            
            us.append(u)

        return us
        

def reconstruct_solutions(
    gbl_dofs_mngr: GlobalDofsManager,
    lambda_: npt.NDArray[np.float64],
    assembler: Assembler
) -> list[dolfinx.fem.Function]:
    """Reconstructs the solution function of every subdomain, starting from the
    multipliers lambda_ at the interfaces (the solution of the global dual
    problem).

    Args:
        gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        lambda_ (npt.NDArray[np.float64]): Solution of the dual problem.
        assembler (Assembler): Fetidp operators manager.

    Returns:
        list[dolfinx.fem.Function]: List of functions describing the solution
            in every single subdomain. The FEM space of every function has the
            same structure as the one of the reference subdomain, but placed
            at its corresponding position.
    """

    uP = assembler.reconstruct_uP(lambda_)
    Urs = assembler.reconstruct_Us(uP, lambda_)

    us = []
    N = gbl_dofs_mngr.get_num_subdomains()
    for s_id in range(N):
        subdomain_i = gbl_dofs_mngr.create_subdomain(s_id)
        uh = dolfinx.fem.Function(subdomain_i.V)
        uh.x.array[:] = Urs[s_id]
        us.append(uh)

    return us


def write_output_subdomains(us: list[dolfinx.fem.Function]) -> None:
    """Writes the solution functions as VTX folders named
    "subdomain_i.pb" into the folder "results", with i running from 0
    to N-1 (N being the number of subdomains). One folder per subdomain.

    To visualize them, the ".pb" folders can be directly imported in
    ParaView.

    Args:
        us (list[dolfinx.fem.Function]): List of functions describing
            the solution in every subdomain.
    """

    for s_id, uh in enumerate(us):
        write_solution(uh, "results", f"subdomain_{s_id}")


def fetidp_solver(n: list[int], N: list[int, int], degree: int) -> None:
    """Solves the Poisson problem with N subdomains per direction using
    a preconditioned conjugate gradient FETI-DP solver.

    The Dirichlet preconditioner is used.

    Every subdomain is considered to have n elements per direction, and
    the input degree is used for discretizing the solution.

    The generated solutions are written to the folder "results" as VTX
    folders named "subdomain_i.pb", with i running from 0 to N-1.
    One file per subdomain. Thy can be visualized using ParaView.

    Args:
        n (list[int]): Number of elements per direction in every single
            subdomain.
        N (list[int]): Number of subdomains per direction.
        degree (int): Discretization space degree.
    """

    assert N[0] * N[1] > 1, "Invalid number of subdomains."
    assert degree > 0, "Invalid degree."

    gbl_dofs_mngr = GlobalDofsManager.create_unit_square(n, degree, N)

    assembler = Assembler(gbl_dofs_mngr)

    check = residual_tracker()

    n_duals = gbl_dofs_mngr.get_num_duals()
    M = LinearOperator((n_duals, n_duals), matvec = assembler.apply_M)
    F = LinearOperator((n_duals, n_duals), matvec = assembler.apply_F)
    dbar = assembler.dbar
    
    lambda_, exit_code = cg(F, dbar, M = M, callback = check)

    if exit_code == 0:
        print("PCG has converged")
        print(f"Number of iterations: {check.niter}")
    else: 
        print("CG has not converged")
        print(f"Number of iterations: {check.niter}")

    us = reconstruct_solutions(gbl_dofs_mngr, lambda_, assembler)
    write_output_subdomains(us)
