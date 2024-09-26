import numpy as np
import numpy.typing as npt
import scipy.sparse

import dolfinx.fem
import ufl

from utils import create_2D_mesh

# Checking version 0.8.0 of dolfinx. Likely, versions 0.8.x are also valid.
assert [int(v) for v in dolfinx.__version__.split(".")] == [0, 8, 0]

type SparseMatrix = scipy.sparse._csr.csr_matrix


class SubDomain:
    def __init__(
        self,
        n: list[int],
        degree: int,
        p0: list[float] = [0.0, 0.0],
        p1: list[float] = [1.0, 1.0],
        assemble: bool = False,
    ):
        """Initializes the subdomain.

        Args:
            n (list[int]): Number of elements per direction in the subdomain.
            degree (int): Discretization space degree.
            p0 (list[float], optional): Bottom-left corner of the subdomain.
                Defaults to [0.0, 0.0].
            p1 (list[float], optional): Top-right corner of the subdomain.
                Defaults to [1.0, 1.0].
            assemble (bool, optional): If True, the stiffness matrix and
                right-hand-side vector are assembled. Defaults to False.
        """
        self.n = n
        self.degree = degree
        self.p0 = p0
        self.p1 = p1
        self.mesh = create_2D_mesh(n, p0, p1)
        self.V = dolfinx.fem.functionspace(self.mesh, ("Lagrange", degree))
        if assemble:
            self.K, self.f = self.create_K_and_f()

    def get_all_dofs(self) -> npt.NDArray[np.int32]:
        """Gets an array with all the degrees-of-freedom in the subdomain's
        space.

        Returns:
            npt.NDArray[np.int32]: Sorted array with all the
                degrees-of-freedom.
        """
        dof_map = self.V.dofmap.index_map
        return np.arange(*dof_map.local_range)

    def get_corners_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the degrees-of-freedom associated to the four courners of the
        subdomain.

        The ordering of the four corners follows the basix convention for
        quadrilaterals.
        See https://docs.fenicsproject.org/basix/v0.8.0/index.html

        Returns:
            npt.NDArray[np.int32]: Sorted array with the corner
            degrees-of-freedom.
        """

        markers = [
            lambda x: np.logical_and(
                np.isclose(x[0], self.p0[0]), np.isclose(x[1], self.p0[1])
            ),
            lambda x: np.logical_and(
                np.isclose(x[0], self.p1[0]), np.isclose(x[1], self.p0[1])
            ),
            lambda x: np.logical_and(
                np.isclose(x[0], self.p0[0]), np.isclose(x[1], self.p1[1])
            ),
            lambda x: np.logical_and(
                np.isclose(x[0], self.p1[0]), np.isclose(x[1], self.p1[1])
            ),
        ]

        corner_dofs = np.empty(4, dtype=np.int32)
        for i in range(4):
            corner_dofs[i] = dolfinx.fem.locate_dofs_geometrical(self.V, markers[i])[0]

        return corner_dofs

    def get_faces_dofs(self) -> list[npt.NDArray[np.int32]]:
        """Gets the degrees-of-freedom for the 4 faces of the subdomain.

        The ordering of the four faces follows the basix convention for
        quadrilaterals.
        See https://docs.fenicsproject.org/basix/v0.8.0/index.html

        Returns:
            list[npt.NDArray[np.int32]]: Sorted arrays of degrees-of-freedom.
                One for every face.
        """
        markers = [
            lambda x: np.isclose(x[1], self.p0[1]),
            lambda x: np.isclose(x[0], self.p0[0]),
            lambda x: np.isclose(x[0], self.p1[0]),
            lambda x: np.isclose(x[1], self.p1[1]),
        ]

        faces_dofs = []
        for i in range(4):
            face_dofs = dolfinx.fem.locate_dofs_geometrical(self.V, markers[i])
            faces_dofs.append(np.sort(face_dofs))

        return faces_dofs

    def get_primal_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the local primal degrees-of-freedom of the subdomain.

        Returns:
            npt.NDArray[np.int32]: Sorted array with the primal
                degrees-of-freedom.
        """

        corner_dofs = self.get_corners_dofs()

        return np.array(corner_dofs)

    def get_remainder_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the local remainder degrees-of-freedom of the subdomain.

        Returns:
            npt.NDArray[np.int32]: Sorted remainder dofs.
        """

        all_dofs = self.get_all_dofs()
        corner_dofs = self.get_corners_dofs()

        rem_dofs = [x for x in all_dofs if x not in corner_dofs]

        return np.array(rem_dofs)

    def get_dual_dofs(
        self,
    ) -> list[npt.NDArray[np.int32]]:
        """Gets the dual degrees-of-freedom of every face of the
        subdomain.

        The ordering of the arrays follows the basix convention for
        faces in quadrilaterals.
        See https://docs.fenicsproject.org/basix/v0.8.0/index.html

        Returns:
            list[npt.NDArray[np.int32]]:
                Dual degrees-of-freedom corresponding to all the faces.
                One sorted and unique array per face.
        """

        faces_dofs = self.get_faces_dofs()
        primal_dofs = self.get_primal_dofs()

        dual_dofs = []
        for face_id in range(4):
            dual_dofs.append(np.setdiff1d(faces_dofs[face_id], primal_dofs))

        return dual_dofs
    
    def get_prec_dofs(
        self,
    ) -> list[npt.NDArray[np.int32]]:
        """Gets the interior and dual degrees-of-freedom of the
        subdomain.

        Returns:
            npt.NDArray[np.int32]:
                Interior degrees-of-freedom. 
                One sorted and unique array.
            npt.NDArray[np.int32]:
                Dual degrees-of-freedom.
                One sorted and unique array.
        """

        interf_dofs = np.hstack(self.get_dual_dofs())
        interf_dofs = np.unique(np.sort(interf_dofs))

        rem_dofs = self.get_remainder_dofs()

        interior_dofs = np.array([x for x in rem_dofs if x not in interf_dofs])

        return interior_dofs, interf_dofs

    def create_K_and_f(self) -> tuple[SparseMatrix, npt.NDArray[np.float64]]:
        """Assembles the stiffness matrix and right-hand-side vector of the
        subdomain.

        Returns:
            tuple[SparseMatrix, npt.NDArray[np.float64]]: Assembled matrix and
                vector.
        """

        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        dx = ufl.dx(domain=self.mesh)
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
        L = dolfinx.fem.Constant(self.mesh, 1.0) * v * ufl.dx

        a_form = dolfinx.fem.forms.form(a)
        L_form = dolfinx.fem.forms.form(L)

        K = dolfinx.fem.assemble_matrix(a_form)
        f = dolfinx.fem.assemble_vector(L_form)

        K = K.to_scipy()
        f = f.array

        return K, f

    def create_Tdr(self) -> SparseMatrix:
        """Creates the restriction operator Tdr from remainder local dofs
        to global dual dofs.

        See paragraph between eqs. (12)-(13) in
            Hirschler, Thibaut, et al. "Reduced order modeling based inexact
            FETI-DP solver for lattice structures."
            International Journal for Numerical Methods in Engineering 125.8 (2024): e7419.

        Returns:
            SparseMatrix: Restriction operator.
        """

        interf_dofs = np.hstack(self.get_dual_dofs())
        interf_dofs = np.unique(np.sort(interf_dofs))

        rem_dofs = self.get_remainder_dofs()

        n_rem = rem_dofs.size
        n_interf = interf_dofs.size

        values = np.ones(n_interf, dtype=np.float64)
        rows = np.arange(n_interf)

        # Assert needed for calling isin
        assert (np.sort(rem_dofs) == rem_dofs).all()
        mask = np.isin(rem_dofs, interf_dofs)
        cols = np.where(mask)[0]

        Tdr = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(n_interf, n_rem))

        return Tdr
