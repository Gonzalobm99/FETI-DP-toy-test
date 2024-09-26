import numpy as np
import numpy.typing as npt
import scipy.sparse

from typing import Self, Callable

import dolfinx.fem

from subdomain import SubDomain
from utils import create_2D_mesh

type SparseMatrix = scipy.sparse._csr.csr_matrix
type Marker = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool]]

# Checking version 0.8.0 of dolfinx. Likely, versions 0.8.x are also valid.
assert [int(v) for v in dolfinx.__version__.split(".")] == [0, 8, 0]


class GlobalDofsManager:
    """Class for managing operations related to the global primal and dual
    degrees-of-freedom of the coarse domain.
    """

    def __init__(self, coarse_mesh: dolfinx.mesh.Mesh, subdomain: SubDomain):
        """Initializes the class.

        Args:
            coarse_mesh (dolfinx.mesh.Mesh): Coarse mesh describing the
                subdomains partition.
            subdomain (SubDomain): Reference subdomain. All the subdomains are
                considered to be equal to this one, but a different positions.
        """

        self.subdomain = subdomain
        self.coarse_mesh = coarse_mesh
        self.primal_V = dolfinx.fem.functionspace(self.coarse_mesh, ("Lagrange", 1))
        self._create_primal_dofs()
        self._create_primal_dofs_Dirichlet()
        self._create_dual_dofs()

    @staticmethod
    def create_unit_square(n: list[int], degree: int, N: list[int]) -> Self:
        """Creates a new GlobalDofsManager in a unit square.

        Args:
            n (list[int]): Number of elements per direction of the reference
                subdomain.
            degree (int): Discretization degree of the reference subdomain.
            N (list[int]): Number of subdomains per direction in the domain.

        Returns:
            Self: Newly generated GlobalDofsManager.
        """
        P0 = [0, 0]
        P1 = [1.0, 1.0]
        coarse_mesh = create_2D_mesh(N, P0, P1)

        p0 = [0, 0]
        p1 = [1.0 / float(N[0]), 1.0 / float(N[1])]
        subdomain = SubDomain(n, degree, p0, p1, assemble=True)
        return GlobalDofsManager(coarse_mesh, subdomain)

    def _create_dual_dofs(self) -> None:
        """Creates the global dual degrees-of-freedom of every facet
        and stores them in self._dual_dofs_ranges.
        """

        top = self.coarse_mesh.topology
        top.create_connectivity(1, 0)
        f_2_v = top.connectivity(1, 0)

        top.create_connectivity(2, 1)  # Needed later on when creating Bd

        x = self.coarse_mesh.geometry.x

        def is_horizontal(facet_id):
            """Returns True if the given facet is horizontal."""
            nodes = f_2_v.links(facet_id)
            x0 = x[nodes[0]]
            x1 = x[nodes[1]]
            return np.abs(x0[0] - x1[0]) > np.abs(x0[1] - x1[1])

        # Number of internal subdomain nodes in each horizontal
        # and vertical interfaces.
        n_int_nodes_in_faces = [
            self.subdomain.n[0] * self.subdomain.degree - 1,
            self.subdomain.n[1] * self.subdomain.degree - 1,
        ]

        top.create_connectivity(1, 2)
        f_2_c = top.connectivity(1, 2)
        n_facets = f_2_c.num_nodes

        self._dual_dofs_ranges = np.empty((n_facets, 2), dtype=np.int32)

        counter = 0
        for facet_id in range(n_facets):
            dir = 0 if is_horizontal(facet_id) else 1

            self._dual_dofs_ranges[facet_id, 0] = counter
            counter += n_int_nodes_in_faces[dir]
            self._dual_dofs_ranges[facet_id, 1] = counter

    def _create_primal_dofs(self) -> None:
        """Creates the global primal degrees-of-freedom of every subdomain
        and stores them in self._primal_dofs.
        """

        n_subs = self.get_num_subdomains()
        self._primal_dofs = np.zeros((n_subs, 4), dtype=np.int32)

        top = self.coarse_mesh.topology
        top.create_connectivity(2, 0)
        c_2_v = top.connectivity(2, 0)

        for s_id in range(n_subs):
            self._primal_dofs[s_id, :] = c_2_v.links(s_id)

    def _create_primal_dofs_Dirichlet(self) -> None:
        """Creates a list of all primal dofs under homogeneous Dirichlet
        conditions and stores it in self._nodes_hom_Dirichlet.
        """

        top = self.coarse_mesh.topology
        top.create_connectivity(1, 0)
        f_2_v = top.connectivity(1, 0)

        top.create_connectivity(1, 2)  # Needed for exterior_facet_indices
        bound_facets = dolfinx.mesh.exterior_facet_indices(top)

        n_facets = bound_facets.size
        self._nodes_hom_Dirichlet = np.empty((n_facets * 2), dtype=np.int32)
        nodes_hom_Dirichlet = []
        for facet_id in bound_facets:
            nodes_hom_Dirichlet.append(f_2_v.links(facet_id))
        nodes_hom_Dirichlet = np.hstack(nodes_hom_Dirichlet)
        self._nodes_hom_Dirichlet = np.unique(np.sort(nodes_hom_Dirichlet))

    def get_active_primal_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the global primal dofs that are not under Dirichlet
        boundary conditions.

        Returns:
            npt.NDArray[np.int32]: Global primal dofs on which Dirichlet
                boundary conditions are not applied. This array is unique and
                sorted.
        """

        all_primal_dofs = np.arange(self.get_num_primals())
        return np.setdiff1d(all_primal_dofs, self._nodes_hom_Dirichlet)

    def create_Ap(self, subdomain_id: int) -> SparseMatrix:
        """Creates the global-local assembly operator Ap for the primal dofs
        of the given subdomain.

        See paragraph between eqs. (17)-(18) in
            Hirschler, Thibaut, et al. "Reduced order modeling based inexact
            FETI-DP solver for lattice structures."
            International Journal for Numerical Methods in Engineering 125.8 (2024): e7419.

        Args:
            subdomain_id (int): Number of subdomain.

        Returns:
            SparseMatrix: Global-local assembly operator.
        """

        assert 0 <= subdomain_id < self.get_num_subdomains()

        n_global_primals = self.get_num_primals()
        n_local_primals = 4

        values = np.ones(n_local_primals, dtype=np.float64)
        rows = self._primal_dofs[subdomain_id]
        cols = np.arange(n_local_primals)

        Ap = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(n_global_primals, n_local_primals))

        return Ap

    def create_Bd(self, subdomain_id: int) -> SparseMatrix:
        """Creates the local (dual) Boolean operator Bd of a given subdomain.

        See paragraph between eqs. (14)-(15) in
            Hirschler, Thibaut, et al. "Reduced order modeling based inexact
            FETI-DP solver for lattice structures."

        Args:
            subdomain_id (int): Number of subdomain.

        Returns:
            SparseMatrix: Local (dual) Boolean operator.
        """

        assert 0 <= subdomain_id < self.get_num_subdomains()

        n_dual = self.get_num_duals()

        local_faces_dual_dofs = self.subdomain.get_dual_dofs()
        all_local_dual_dofs = np.hstack(local_faces_dual_dofs)
        n_local_duals = all_local_dual_dofs.size

        sort_ids = np.argsort(all_local_dual_dofs)

        values = np.ones(n_local_duals, dtype=np.float64)
        values[int(n_local_duals / 2) :] = -1.0
        values = values[sort_ids]

        c_2_f = self.coarse_mesh.topology.connectivity(2, 1)
        facets = c_2_f.links(subdomain_id)

        global_dual_dofs = []
        for start, end in self._dual_dofs_ranges[facets]:
            global_dual_dofs.append(np.arange(start, end))
        rows = np.hstack(global_dual_dofs)
        rows = rows[sort_ids]

        cols = np.arange(n_local_duals)

        Bd = scipy.sparse.csr_matrix(
            (values, (rows, cols)), shape=(n_dual, n_local_duals)
        )
        return Bd

    def get_num_subdomains(self) -> int:
        """Gets the number of subdomains.

        Returns:
            int: Number of subdomains.
        """
        return self.coarse_mesh.geometry.dofmap.shape[0]

    def get_num_primals(self) -> int:
        """Gets the number of global primal degrees-of-freedom.

        Returns:
            int: Number of primal degrees-of-freedom.
        """
        return self.primal_V.dofmap.index_map.size_local

    def get_num_duals(self) -> int:
        """Gets the number of global dual degrees-of-freedom.

        Returns:
            int: Number of dual degrees-of-freedom.
        """
        return self._dual_dofs_ranges[-1, 1]

    def create_subdomain(self, subdomain_id: int) -> SubDomain:
        """Creates a new subdomain with the same properties
        as the reference one, but a new position corresponding
        to the one of the given subdomain.

        Stiffness and right-hand-side operators are not precomputed
        in the new subdomain.

        Args:
            subdomain_id (int): Id of the subdomain to be created.

        Returns:
            SubDomain: Newly created subdomain.
        """

        verts = self.coarse_mesh.geometry.dofmap[subdomain_id]
        x = self.coarse_mesh.geometry.x
        p0 = x[verts[0]]
        p1 = x[verts[-1]]

        n = self.subdomain.n
        degree = self.subdomain.degree

        return SubDomain(n, degree, p0, p1, assemble=False)
