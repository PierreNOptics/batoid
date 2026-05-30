from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np

from . import _batoid
from .trace import intersect, rSplit, reflect, refract, refractScreen


class Surface(ABC):
    """Abstract base class representing a 2D geometric surface.
    """
    def sag(self, x, y):
        """The function defining the surface; z(x, y).

        Parameters
        ----------
        x, y : array_like, shape (n,)
            Positions at which to evaluate the surface sag.

        Returns
        -------
        z : array_like, shape (n,)
            Surface height.
        """
        return self._surface.sag(x, y)

    def normal(self, x, y):
        """The normal vector to the surface at (x, y, z(x, y)).

        Parameters
        ----------
        x, y : array_like, shape (n,)
            Positions at which to evaluate the surface normal.

        Returns
        -------
        normal : array_like, shape (n, 3)
            Surface normals.
        """
        xx = np.asfortranarray(x, dtype=float)
        yy = np.asfortranarray(y, dtype=float)
        out = np.empty(xx.shape+(3,), order='F', dtype=float)
        size = len(xx.ravel())

        self._surface.normal(
            xx.ctypes.data, yy.ctypes.data, size, out.ctypes.data
        )
        try:
            len(x)
        except TypeError:
            return out[0]
        else:
            return out

    def intersect(self, rv, coordSys=None, coating=None):
        return intersect(self, rv, coordSys, coating)

    def reflect(self, rv, coordSys=None, coating=None):
        """Calculate intersection of rays with this surface, and immediately
        reflect the rays at the points of intersection.

        Parameters
        ----------
        rv : RayVector
            Rays to reflect.
        coordSys : CoordSys, optional
            If present, then use for the coordinate system of the surface.  If
            ``None`` (default), then assume that rays and surface are already
            expressed in the same coordinate system.
        coating : Coating, optional
            Apply this coating upon surface intersection.

        Returns
        -------
        outRays : RayVector
            New object corresponding to original rays propagated and reflected.
        """
        return reflect(self, rv, coordSys, coating)

    def refract(self, rv, inMedium, outMedium, coordSys=None, coating=None):
        """Calculate intersection of rays with this surface, and immediately
        refract the rays through the surface at the points of intersection.

        Parameters
        ----------
        rv : RayVector
            Rays to refract.
        inMedium : Medium
            Refractive medium on the incoming side of the surface.
        outMedium : Medium
            Refractive medium on the outgoing side of the surface.
        coordSys : CoordSys, optional
            If present, then use for the coordinate system of the surface.  If
            ``None`` (default), then assume that rays and surface are already
            expressed in the same coordinate system.
        coating : Coating, optional
            Apply this coating upon surface intersection.

        Returns
        -------
        outRays : RayVector
            New object corresponding to original rays propagated and refracted.
        """
        return refract(self, rv, inMedium, outMedium, coordSys, coating)

    def rSplit(self, rv, inMedium, outMedium, coating, coordSys=None):
        """Calculate intersection of rays with this surface, and immediately
        split the rays into reflected and refracted rays, with appropriate
        fluxes.

        Parameters
        ----------
        rv : RayVector
            Rays to refract.
        inMedium : Medium
            Refractive medium on the incoming side of the surface.
        outMedium : Medium
            Refractive medium on the outgoing side of the surface.
        coating : Coating
            Coating object to control transmission coefficient.
        coordSys : CoordSys, optional
            If present, then use for the coordinate system of the surface.  If
            ``None`` (default), then assume that rays and surface are already
            expressed in the same coordinate system.

        Returns
        -------
        reflectedRays, refractedRays : RayVector
            New objects corresponding to original rays propagated and
            reflected/refracted.
        """
        return rSplit(self, rv, inMedium, outMedium, coating, coordSys)

    def refractScreen(self, rv, screen, coordSys=None):
        """Calculate intersection of rays with this surface, and immediately
        refract the rays through the phase screen at the points of intersection.

        Parameters
        ----------
        rv : RayVector
            Rays to refract.
        screen : Surface
            OPD to add to rays (in meters) as they cross this interface.
        coordSys : CoordSys, optional
            If present, then use for the coordinate system of the surface.  If
            ``None`` (default), then assume that rays and surface are already
            expressed in the same coordinate system.

        Returns
        -------
        outRays : RayVector
            New object corresponding to original rays propagated and refracted.
        """
        return refractScreen(self, rv, screen, coordSys)

    def __ne__(self, rhs):
        return not (self == rhs)

    def __add__(self, rhs):
        return Sum(self, rhs)


class Plane(Surface):
    """Planar surface.  The surface sag follows the equation:

    .. math::

        z(x, y) = 0
    """
    def __init__(self):
        self._surface = _batoid.CPPPlane()

    def __hash__(self):
        return hash("batoid.Plane")

    def __setstate__(self, tup):
        self.__init__()

    def __getstate__(self):
        return ()

    def __eq__(self, rhs):
        return isinstance(rhs, Plane)

    def __repr__(self):
            return "Plane()"


class Tilted(Surface):
    """Tilted planar surface.  The surface sag follows the equation:

    .. math::

        z(x, y) = x \\tan(\\theta_x) + y \\tan(\\theta_y)
    """
    def __init__(self, tanx, tany):
        self.tanx = tanx
        self.tany = tany
        self._surface = _batoid.CPPTilted(tanx, tany)

    def __hash__(self):
        return hash(("batoid.Tilted", self.tanx, self.tany))

    def __setstate__(self, tup):
        tanx, tany = tup
        self.__init__(tanx, tany)

    def __getstate__(self):
        return (self.tanx, self.tany)

    def __eq__(self, rhs):
        if not isinstance(rhs, Tilted): return False
        return self.tanx == rhs.tanx and self.tany == rhs.tany

    def __repr__(self):
            return f"Tilted({self.tanx}, {self.tany})"


class Paraboloid(Surface):
    """Surface of revolution with parabolic cross-section, and where the axis
    of revolution is along the axis of the parabola.  The surface sag follows
    the equation

    .. math::

        z(x, y) = z(r) = \\frac{r^2}{2 R}

    where :math:`r = \\sqrt{x^2 + y^2}` and ``R`` is the radius of curvature at
    the paraboloid vertex.

    Parameters
    ----------
    R : float
        Radius of curvature at paraboloid vertex.
    """
    def __init__(self, R):
        self.R = R
        self._surface = _batoid.CPPParaboloid(R)

    def __hash__(self):
        return hash(("batoid.Paraboloid", self.R))

    def __setstate__(self, R):
        self.__init__(R)

    def __getstate__(self):
        return self.R

    def __eq__(self, rhs):
        if not isinstance(rhs, Paraboloid): return False
        return self.R == rhs.R

    def __repr__(self):
        return f"Paraboloid({self.R})"


class Sphere(Surface):
    """Spherical surface.  The surface sag follows the equation:

    .. math::

        z(x, y) = z(r) = R \\left(1 - \\sqrt{1-\\frac{r^2}{R^2}}\\right)

    where :math:`r = \\sqrt{x^2 + y^2}` and ``R`` is the radius the sphere.
    Note that the center of the sphere is a distance ``R`` above the vertex.

    Parameters
    ----------
    R : float
        Sphere radius.
    """
    def __init__(self, R):
        self.R = R
        self._surface = _batoid.CPPSphere(R)

    def __hash__(self):
        return hash(("batoid.Sphere", self.R))

    def __setstate__(self, R):
        self.__init__(R)

    def __getstate__(self):
        return self.R

    def __eq__(self, rhs):
        if not isinstance(rhs, Sphere): return False
        return self.R == rhs.R

    def __repr__(self):
        return f"Sphere({self.R})"


class Quadric(Surface):
    """Surface of revolution where the cross section is a conic section.
    The surface sag follows the equation:

    .. math::

        z(x, y) = z(r) = \\frac{r^2}{R \\left(1 + \\sqrt{1 - \\frac{r^2}{R^2} (1 + \\kappa)}\\right)}

    where :math:`r = \\sqrt{x^2 + y^2}`, ``R`` is the radius of curvature at the
    surface vertex, and :math:`\\kappa` is the conic constant.  Different
    ranges of :math:`\\kappa` indicate different categories of surfaces:

        - :math:`\\kappa > 0`      =>  oblate ellipsoid
        - :math:`\\kappa = 0`      =>  sphere
        - :math:`-1 < \\kappa < 0` =>  prolate ellipsoid
        - :math:`\\kappa = -1`    =>  paraboloid
        - :math:`\\kappa < -1`     =>  hyperboloid

    Parameters
    ----------
    R : float
        Radius of curvature at vertex.
    conic : float
        Conic constant :math:`\\kappa`
    """
    def __init__(self, R, conic):
        self.R = R
        self.conic = conic
        self._surface = _batoid.CPPQuadric(R, conic)

    def __hash__(self):
        return hash(("batoid.Quadric", self.R, self.conic))

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return (self.R, self.conic)

    def __eq__(self, rhs):
        if not isinstance(rhs, Quadric): return False
        return (self.R == rhs.R and
                self.conic == rhs.conic)

    def __repr__(self):
        return f"Quadric({self.R}, {self.conic})"


class Asphere(Surface):
    """Surface of revolution where the cross section is a conic section plus an
    even polynomial.  Represents the equation

    The surface sag follows the equation:

    .. math::

        z(x, y) = z(r) = \\frac{r^2}{R \\left(1 + \\sqrt{1 - \\frac{r^2}{R^2} (1 + \\kappa)}\\right)} + \\sum_i \\alpha_i r^{2 i}

    where :math:`r = \\sqrt{x^2 + y^2}`, ``R`` is the radius of curvature at the
    surface vertex, :math:`\\kappa` is the conic constant, and
    :math:`\\left\\{\\alpha_i\\right\\}` are the even polynomial coefficients.
    Different ranges of :math:`\\kappa` produce different categories of
    surfaces (where alpha==0):

        - :math:`\\kappa > 0`      =>  oblate ellipsoid
        - :math:`\\kappa = 0`      =>  sphere
        - :math:`-1 < \\kappa < 0` =>  prolate ellipsoid
        - :math:`\\kappa = -1`    =>  paraboloid
        - :math:`\\kappa < -1`     =>  hyperboloid

    Parameters
    ----------
    R : float
        Radius of curvature at vertex.
    conic : float
        Conic constant :math:`\\kappa`.
    coefs : list of float
        Even polynomial coefficients :math:`\\left\\{\\alpha_i\\right\\}`
    imin : int, optional
        Starting index for polynomial coefficients.  The default value of 2
        means that coefs[0] is the coefficient of :math:`r^4` and coefs[1] is
        the coefficient of :math:`r^6`.  To enable a coefficient of :math:`r^2`
        set ``imin=1``.
    """
    def __init__(self, R, conic, coefs, imin=2):
        self.R = R
        self.conic = conic
        self.coefs = coefs
        self.imin = imin

        if not isinstance(imin, Integral):
            raise TypeError("imin must be an int")
        if imin <= 0:
            raise ValueError("imin must be >= 1")
        if not all(isinstance(coef, Real) for coef in coefs):
            raise TypeError("coefs must be a list of floats")

        # padded_coefs starts at i=1, so the 0th term is a_1 r^2.
        if imin != 1:
            padded_coefs = np.concatenate([np.zeros(imin-1), coefs])
        else:
            padded_coefs = coefs
        self._padded_coefs = np.ascontiguousarray(padded_coefs)
        self._surface = _batoid.CPPAsphere(
            R, conic, self._padded_coefs.ctypes.data, len(self._padded_coefs)
        )

    def __hash__(self):
        return hash((
            "batoid.Asphere",
            self.R,
            self.conic,
            tuple(self.coefs),
            self.imin
        ))

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return self.R, self.conic, self.coefs, self.imin

    def __eq__(self, rhs):
        if not isinstance(rhs, Asphere): return False
        return (self.R == rhs.R and
                self.conic == rhs.conic and
                np.array_equal(self.coefs, rhs.coefs) and
                self.imin == rhs.imin
               )

    def __repr__(self):
        out = f"Asphere({self.R}, {self.conic}, {self.coefs!r}"
        if self.imin == 2:
            out += ")"
        else:
            out += f", imin={self.imin})"
        return out
    
class Biconic(Surface):
    """Biconic surface where the curvature and conic constant can be different 
    along the x and y axes. The surface sag follows the equation:

    .. math::

        z(x, y) = \\frac{c_x x^2}{1 + \\sqrt{1 - (1 + k_x) c_x^2 x^2}} + 
                  \\frac{c_y y^2}{1 + \\sqrt{1 - (1 + k_y) c_y^2 y^2}}

    where:

    - :math:`c_x = \\frac{1}{R_x}` is the curvature along the X-axis.
    - :math:`c_y = \\frac{1}{R_y}` is the curvature along the Y-axis.
    - :math:`k_x` and :math:`k_y` are the conic constants in the X and Y directions.

    Different ranges of :math:`k_x` and :math:`k_y` indicate different categories 
    of surfaces in each axis.

    Parameters
    ----------
    Rx : float
        Radius of curvature in the x-direction.
    Ry : float
        Radius of curvature in the y-direction.
    kx : float
        Conic constant in the x-direction.
    ky : float
        Conic constant in the y-direction.
    """

    def __init__(self, Rx, Ry, kx, ky):
        self.Rx = Rx
        self.Ry = Ry
        self.kx = kx
        self.ky = ky
        self._surface = _batoid.CPPBiconic(Rx, Ry, kx, ky)

    def __hash__(self):
        return hash(("batoid.Biconic", self.Rx, self.Ry, self.kx, self.ky))

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return (self.Rx, self.Ry, self.kx, self.ky)

    def __eq__(self, rhs):
        if not isinstance(rhs, Biconic):
            return False
        return (self.Rx == rhs.Rx and
                self.Ry == rhs.Ry and
                self.kx == rhs.kx and
                self.ky == rhs.ky)

    def __repr__(self):
        return f"Biconic({self.Rx}, {self.Ry}, {self.kx}, {self.ky})"

class Zernike(Surface):
    """Surface defined by Zernike polynomials.  The surface sag follows the
    equation:

    .. math::

        z(x, y) = \\sum_j a_j Z_j\\left(\\epsilon; \\frac{x}{R_{outer}}, \\frac{y}{R_{outer}}\\right)

    where :math:`Z_j(\\epsilon, u, v)` are the annular Zernike polynomials
    (Mahajan) with central obscuration
    :math:`\\epsilon = \\frac{R_{inner}}{R_{outer}}`
    indexed by the Noll (1976) convention, :math:`R_{outer}` is the outer
    radius of the annulus, :math:`R_{inner}` is the inner radius of the
    annulus, and :math:`\\left\\{a_j\\right\\}` are the annular
    coefficients.

    Note that the Noll convention starts at j=1, so the :math:`a_0` =
    ``coef[0]`` value has no effect on the surface.

    Parameters
    ----------
    coef : list of float
        Annular Zernike polynomial coefficients.
    R_outer : float, optional
        Outer radius of annulus.
    R_inner : float, optional
        Inner radius of annulus.
    x_origin, y_origin : float, optional
        Origin of the Zernike polynomial.  Default is (0, 0).
    """
    def __init__(self, coef, R_outer=1.0, R_inner=0.0, x_origin=0.0, y_origin=0.0):
        import galsim

        self.coef = np.array(coef, dtype=float, order="C")
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)
        self.x_origin = float(x_origin)
        self.y_origin = float(y_origin)
        self.Z = galsim.zernike.Zernike(coef, R_outer, R_inner)
        self._xycoef = self.Z._coef_array_xy
        self._xycoef_gradx = self.Z.gradX._coef_array_xy
        self._xycoef_grady = self.Z.gradY._coef_array_xy

        self._surface = _batoid.CPPPolynomialSurface(
            self._xycoef.ctypes.data,
            self._xycoef_gradx.ctypes.data,
            self._xycoef_grady.ctypes.data,
            self.x_origin, self.y_origin,
            self._xycoef.shape[0],
            self._xycoef.shape[1]
        )

    def __hash__(self):
        return hash((
            "batoid.Zernike",
            tuple(self.coef),
            self.R_outer, self.R_inner,
            self.x_origin, self.y_origin,
        ))

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return self.coef, self.R_outer, self.R_inner, self.x_origin, self.y_origin

    def __eq__(self, rhs):
        if not isinstance(rhs, Zernike): return False
        return (np.array_equal(self.coef, rhs.coef) and
                self.R_outer == rhs.R_outer and
                self.R_inner == rhs.R_inner and
                self.x_origin == rhs.x_origin and
                self.y_origin == rhs.y_origin)

    def __repr__(self):
        out = f"Zernike({self.coef!r}"
        if self.R_outer != 1.0:
            out += f", R_outer={self.R_outer}"
        if self.R_inner != 0.0:
            out += f", R_inner={self.R_inner}"
        if self.x_origin != 0.0:
            out += f", x_origin={self.x_origin}"
        if self.y_origin != 0.0:
            out += f", y_origin={self.y_origin}"
        out += ")"
        return out


class Bicubic(Surface):
    """Surface defined by interpolating from a grid.

    Parameters
    ----------
    xs, ys : array_like
        1d uniform-spaced arrays indicating the grid points.
    zs : array_like
        2d array indicating the surface.
    dzdxs : array_like, optional
        2d array indicating derivatives dz/dx at grid points.
    dzdys : array_like, optional
        2d array indicating derivatives dz/dy at grid points.
    d2zdxdys : array_like, optional
        2d array indicating mixed derivatives d^2 z / (dx dy) at grid points.
    nanpolicy : {'zero', 'nan'}
        Return zero or nan for requests outside input domain?
    """
    def __init__(
        self, xs, ys, zs, dzdxs=None, dzdys=None, d2zdxdys=None, nanpolicy='nan'
    ):
        assert nanpolicy.upper() in ['NAN', 'ZERO']
        self._xs = np.array(xs, dtype=float, order="C")
        self._ys = np.array(ys, dtype=float, order="C")
        self._zs = np.array(zs, dtype=float, order="C")
        self._x0 = xs[0]
        self._y0 = ys[0]
        dx = self._dx = (self._xs[-1] - self._xs[0])/(len(self._xs)-1)
        dy = self._dy = (self._ys[-1] - self._ys[0])/(len(self._ys)-1)

        if dzdxs is None:
            dzdxs = np.empty_like(self._zs)
            dzdxs[:, 1:-1] = (self._zs[:, 2:] - self._zs[:, :-2])/(2*dx)
            dzdxs[:, 0] = (self._zs[:, 1] - self._zs[:, 0])/dx
            dzdxs[:, -1] = (self._zs[:, -1] - self._zs[:, -2])/dx

        if dzdys is None:
            dzdys = np.empty_like(self._zs)
            dzdys[1:-1, :] = (self._zs[2:, :] - self._zs[:-2, :])/(2*dy)
            dzdys[0, :] = (self._zs[1, :] - self._zs[0, :])/dy
            dzdys[-1, :] = (self._zs[-1, :] - self._zs[-2, :])/dy

        if d2zdxdys is None:
            d2zdxdys = np.empty_like(self._zs)
            d2zdxdys[:, 1:-1] = (dzdys[:, 2:] - dzdys[:, :-2])/(2*dx)
            d2zdxdys[:, 0] = (dzdys[:, 1] - dzdys[:, 0])/dx
            d2zdxdys[:, -1] = (dzdys[:, -1] - dzdys[:, -2])/dx

        self._dzdxs = np.array(dzdxs, dtype=float, order="C")
        self._dzdys = np.array(dzdys, dtype=float, order="C")
        self._d2zdxdys = np.array(d2zdxdys, dtype=float, order="C")

        self.nanpolicy = nanpolicy
        self._table = _batoid.CPPTable(
            self._x0, self._y0, self._dx, self._dy,
            self._zs.ctypes.data,
            self._dzdxs.ctypes.data,
            self._dzdys.ctypes.data,
            self._d2zdxdys.ctypes.data,
            len(self._xs),
            len(self._ys),
            True if self.nanpolicy.upper() == 'NAN' else False
        )
        self._surface = _batoid.CPPBicubic(self._table)

    @property
    def xs(self):
        return self._xs

    @property
    def ys(self):
        return self._ys

    @property
    def zs(self):
        return self._zs

    @property
    def dzdxs(self):
        return self._dzdxs

    @property
    def dzdys(self):
        return self._dzdys

    @property
    def d2zdxdys(self):
        return self._d2zdxdys

    def __hash__(self):
        return hash((
            "Bicubic", tuple(self.xs), tuple(self.ys), tuple(self.zs.ravel()),
            tuple(self.dzdxs.ravel()), tuple(self.dzdys.ravel()),
            tuple(self.d2zdxdys.ravel()), self.nanpolicy
        ))

    def __setstate__(self, args):
        (self._xs, self._ys, self._zs,
         self._dzdxs, self._dzdys, self._d2zdxdys, self.nanpolicy
        ) = args
        self._x0 = self._xs[0]
        self._y0 = self._ys[0]
        self._dx = (self._xs[-1] - self._xs[0])/(len(self._xs)-1)
        self._dy = (self._ys[-1] - self._ys[0])/(len(self._ys)-1)

        self._table = _batoid.CPPTable(
            self._x0, self._y0, self._dx, self._dy,
            self._zs.ctypes.data,
            self._dzdxs.ctypes.data,
            self._dzdys.ctypes.data,
            self._d2zdxdys.ctypes.data,
            len(self._xs),
            len(self._ys),
            True if self.nanpolicy.upper() == 'NAN' else False
        )
        self._surface = _batoid.CPPBicubic(self._table)

    def __getstate__(self):
        return (
            self.xs, self.ys, self.zs,
            self.dzdxs, self.dzdys, self.d2zdxdys, self.nanpolicy
        )

    def __eq__(self, rhs):
        if not isinstance(rhs, Bicubic): return False
        return (
            np.array_equal(self.xs, rhs.xs)
            and np.array_equal(self.ys, rhs.ys)
            and np.array_equal(self.zs, rhs.zs)
            and np.array_equal(self.dzdxs, rhs.dzdxs)
            and np.array_equal(self.dzdys, rhs.dzdys)
            and np.array_equal(self.d2zdxdys, rhs.d2zdxdys)
            and self.nanpolicy.upper() == rhs.nanpolicy.upper()
        )

    def __repr__(self):
        out = f"Bicubic({self.xs!r}, {self.ys!r}, {self.zs!r}, "
        out += f"{self.dzdxs!r}, {self.dzdys!r}, {self.d2zdxdys!r}"
        if self.nanpolicy.upper() == "NAN":
            out += ")"
        else:
            out += ", nanpolicy='zero')"
        return out


class Sum(Surface):
    """Composite surface combining two or more other Surfaces through addition.
    The surface sag follows the equation:

    .. math::

        z(x, y) = \\sum_i S_i(x, y)

    where :math:`S_i` is the ith input `Surface`.

    Note that Sum-Ray intersection calculations will use the intersection of the
    ray with the first surface in the list as an initial guess for the
    intersection with the full Sum surface.  Thus it is usually a good idea to
    place any surface with an analytic intersection (Quadric or simpler) first
    in the list, and any small perturbations around that surface after.

    Parameters
    ----------
    surfaces : list of Surface
        `Surface` s to add together.
    """
    def __init__(self, *args):
        from collections.abc import Sequence
        if len(args) == 1 and isinstance(args[0], Sequence):
            args = args[0]
        assert all(isinstance(arg, Surface) for arg in args)

        self.surfaces = tuple(args)
        self._surface = _batoid.CPPSum([s._surface for s in self.surfaces])

    def __hash__(self):
        return hash(("batoid.Sum", tuple(self.surfaces)))

    def __setstate__(self, surfaces):
        self.__init__(surfaces)

    def __getstate__(self):
        return self.surfaces

    def __eq__(self, rhs):
        if not isinstance(rhs, Sum): return False
        return self.surfaces == rhs.surfaces  # order matters!

    def __repr__(self):
        return f"Sum({self.surfaces})"
    
# class _ExtendedPolynomials:
#     """Surface defined by an extended polynomial basis. The surface sag follows the equation:

#     .. math::

#         z(x, y) = \sum_{i=1}^N A_i E_i\left(\frac{x}{R_{\mathrm{norm}}}, \frac{y}{R_{\mathrm{norm}}}\right)

#     where :math:`E_i(x, y) = \sum_{k=0}^i x^{i-k} y^k` is the i-th basis polynomial,
#     :math:`A_i` are the polynomial coefficients, and :math:`R_{\mathrm{norm}}` is the normalization radius.

#     The basis is ordered by total degree, omitting the piston term (i=0).
#     Fast evaluation is performed using NumPy arrays.
#     """
#     _monomial_cache = {}

#     @classmethod
#     def _get_monomial_indices(cls, maxdeg):
#         """Return the (px, py) exponents for all monomials of total degree 1 to maxdeg.

#         Uses a class-level cache to avoid recomputation for repeated calls with the same maxdeg.

#         The returned array has shape (n_terms, 2), where each row gives the exponents (px, py)
#         for the monomial x**px * y**py, ordered by increasing total degree and omitting the piston (degree 0).

#         Parameters
#         ----------
#         maxdeg : int
#             Maximum total degree of the monomials (excluding the piston term).

#         Returns
#         -------
#         arr : ndarray of shape (n_terms, 2)
#             Array of exponents for each monomial term.
#         """
#         if maxdeg not in cls._monomial_cache:
#             # Number of monomials (excluding piston)
#             n_terms = maxdeg * (maxdeg + 3) // 2
#             arr = np.empty((n_terms, 2), dtype=int)
#             idx = 0
#             for d in range(1, maxdeg+1):
#                 for px in range(d, -1, -1):
#                     py = d - px
#                     arr[idx, 0] = px
#                     arr[idx, 1] = py
#                     idx += 1
#             cls._monomial_cache[maxdeg] = arr
#         return cls._monomial_cache[maxdeg]
    
#     def __init__(self, A, R_norm=1.0):
#         self.A = np.asarray(A, dtype=float, order="C")
#         self.N = len(A) - 1
#         self.R_norm = float(R_norm)
#         self._maxdeg = self.N
#         self._coef_array_xy = np.zeros((self.N + 1, self.N + 1), dtype=float, order="C")

#         monomials = self._get_monomial_indices(self._maxdeg)
#         n_terms = min(len(monomials), len(A)-1) # for safety
#         pxs, pys = monomials[:n_terms, 0], monomials[:n_terms, 1]

#         scales = np.power(self.R_norm, -(pxs + pys)) # Scale for normalized coordinates
#         self._coef_array_xy[pxs, pys] += self.A[1:1+n_terms] * scales

#         # Precompute gradient arrays
#         self._coef_array_xy_gradx = np.zeros((self.N, self.N), dtype=float, order="C")
#         self._coef_array_xy_grady = np.zeros((self.N, self.N), dtype=float, order="C")
#         # For gradx: d/dx(x^p * y^q) = p * x^{p-1} * y^q
#         px = np.arange(1, self.N + 1)[:, None]
#         self._coef_array_xy_gradx[:, :] = px * self._coef_array_xy[1:, :-1]
#         # For grady: d/dy(x^p * y^q) = q * x^p * y^{q-1}
#         py = np.arange(1, self.N + 1)
#         self._coef_array_xy_grady[:, :] = self._coef_array_xy[:-1, 1:] * py

#     def _prepare_powers(self, x, y, coef_array):
#         """
#         Convert x, y to arrays and compute all powers up to the shape of coef_array.
#         Returns (x, y, x_powers, y_powers)
#         """
#         x, y = np.asarray(x), np.asarray(y)
#         deg_x, deg_y = coef_array.shape
#         x_powers = np.array([x**i for i in range(deg_x)])
#         y_powers = np.array([y**j for j in range(deg_y)])
#         return x, y, x_powers, y_powers

#     def eval(self, x, y):
#         """
#         Evaluate the value of the polynomial surface at (x, y).

#         Uses np.tensordot to efficiently compute the sum over all monomial terms:
#             eval(x, y) = sum_{p, q} coef[p, q] * x**p * y**q
#         where coef contains the precomputed coefficients for each monomial.
#         The first tensordot contracts over the x powers, and the second over the y powers,
#         resulting in the full surface value(s) at the input coordinates.

#         Parameters
#         ----------
#         x, y : array_like
#             Points at which to evaluate the surface.

#         Returns
#         -------
#         Z : array_like
#             The value(s) of the surface at the given (x, y).
#         """
#         x, y, x_powers, y_powers = self._prepare_powers(x, y, self._coef_array_xy)
#         Z = np.tensordot(self._coef_array_xy, x_powers, axes=(0,0))
#         Z = np.tensordot(Z, y_powers, axes=(0,0))
#         return Z

#     def gradx(self, x, y):
#         """
#         Evaluate the partial derivative of the polynomial surface with respect to x at (x, y).

#         Uses np.tensordot to efficiently compute the sum over all monomial terms:
#             gradx(x, y) = sum_{p, q} coef_gradx[p, q] * x**p * y**q
#         where coef_gradx contains the precomputed coefficients for the x-derivative.
#         The first tensordot contracts over the x powers, and the second over the y powers,
#         resulting in the full gradient value(s) at the input coordinates.

#         Parameters
#         ----------
#         x, y : array_like
#             Points at which to evaluate the x-derivative of the surface.

#         Returns
#         -------
#         Zx : array_like
#             The value(s) of the partial derivative with respect to x at the given (x, y).
#         """
#         x, y, x_powers, y_powers = self._prepare_powers(x, y, self._coef_array_xy_gradx)
#         Zx = np.tensordot(self._coef_array_xy_gradx, x_powers, axes=(0,0))
#         Zx = np.tensordot(Zx, y_powers, axes=(0,0))
#         return Zx

#     def grady(self, x, y):
#         """
#         Evaluate the partial derivative of the polynomial surface with respect to y at (x, y).

#         Uses np.tensordot to efficiently compute the sum over all monomial terms:
#             grady(x, y) = sum_{p, q} coef_grady[p, q] * x**p * y**q
#         where coef_grady contains the precomputed coefficients for the y-derivative.
#         The first tensordot contracts over the x powers, and the second over the y powers,
#         resulting in the full gradient value(s) at the input coordinates.

#         Parameters
#         ----------
#         x, y : array_like
#             Points at which to evaluate the y-derivative of the surface.

#         Returns
#         -------
#         Zy : array_like
#             The value(s) of the partial derivative with respect to y at the given (x, y).
#         """
#         x, y, x_powers, y_powers = self._prepare_powers(x, y, self._coef_array_xy_grady)
#         Zy = np.tensordot(self._coef_array_xy_grady, x_powers, axes=(0,0))
#         Zy = np.tensordot(Zy, y_powers, axes=(0,0))
#         return Zy
    
# class XPolynom(Surface):
#     """Surface defined by a 2D polynomial expansion in $x$ and $y$ coordinates.

#     The surface sag follows the equation:

#     .. math::

#         z(x, y) = \sum_{i=1}^N A_i E_i\left(\frac{x-x_0}{R_{\mathrm{norm}}}, \frac{y-y_0}{R_{\mathrm{norm}}}\right)

#     where :math:`E_i(x, y) = \sum_{k=0}^i x^{i-k} y^k` is the i-th basis polynomial,
#     :math:`A_i` are the polynomial coefficients, :math:`R_{\mathrm{norm}}` is the normalization radius,
#     and :math:`(x_0, y_0)` is the origin for the polynomial expansion.

#     The basis is ordered by total degree, omitting the piston term (i=0).
#     Fast evaluation and derivatives are performed using precomputed NumPy arrays.

#     Parameters
#     ----------
#     coef : list of float
#         Polynomial coefficients $A_i$ (excluding piston).
#     R_norm : float, optional
#         Normalization radius for $x$ and $y$ coordinates. Default is 1.0.
#     x_origin, y_origin : float, optional
#         Origin for the polynomial expansion. Default is (0, 0).
#     """
#     def __init__(self, coef, R_norm=1.0, x_origin=0.0, y_origin=0.0):
#         # Store polynomial coefficients and parameters
#         self.coef = np.array(coef, dtype=float, order="C")
#         self.R_norm = float(R_norm)
#         self.x_origin = float(x_origin)
#         self.y_origin = float(y_origin)

#         # Build the polynomial representation (prepend 0 for piston)
#         self.P = _ExtendedPolynomials([0.0] + list(coef), R_norm=self.R_norm)

#         # Precompute coefficient arrays for evaluation and gradients
#         self._xycoef = self.P._coef_array_xy
#         self._xycoef_gradx = self.P._coef_array_xy_gradx
#         self._xycoef_grady = self.P._coef_array_xy_grady

#         # Create the underlying C++ polynomial surface for fast evaluation
#         self._surface = _batoid.CPPPolynomialSurface(
#             self._xycoef.ctypes.data,
#             self._xycoef_gradx.ctypes.data,
#             self._xycoef_grady.ctypes.data,
#             self.x_origin, self.y_origin,
#             self._xycoef.shape[0],
#             self._xycoef.shape[1]
#         )

#     def __hash__(self):
#         return hash((
#             "batoid.XPolynom",
#             tuple(self.coef),
#             self.R_norm,
#             self.x_origin,
#             self.y_origin,
#         ))

#     def __setstate__(self, args):
#         self.__init__(*args)

#     def __getstate__(self):
#         return (self.coef, self.R_norm, self.x_origin, self.y_origin)

#     def __eq__(self, rhs):
#         if not isinstance(rhs, XPolynom):
#             return False
#         return (np.array_equal(self.coef, rhs.coef) and
#                 self.R_norm == rhs.R_norm and
#                 self.x_origin == rhs.x_origin and
#                 self.y_origin == rhs.y_origin)

#     def __repr__(self):
#         out = f"XPolynom({self.coef!r}"
#         if self.R_norm != 1.0:
#             out += f", R_norm={self.R_norm}"
#         if self.x_origin != 0.0:
#             out += f", x_origin={self.x_origin}"
#         if self.y_origin != 0.0:
#             out += f", y_origin={self.y_origin}"
#         out += ")"
#         return out

class _ExtendedPolynomials:
    """Surface defined by an extended polynomial basis. The surface sag follows the equation:

    .. math::

        z(x, y) = \sum_{i=1}^N A_i m_i\left(\frac{x}{R_{\mathrm{norm}}}, \frac{y}{R_{\mathrm{norm}}}\right)

    where :math:`m_i(x, y) = x^{p_i} y^{q_i}` is the i-th individual monomial with exponents
    :math:`(p_i, q_i)` ordered by increasing total degree :math:`p_i + q_i`, and for equal
    total degree by decreasing x-power (i.e., x, y, x^2, xy, y^2, x^3, ...),
    :math:`A_i` are the polynomial coefficients, and :math:`R_{\mathrm{norm}}` is the normalization radius.

    The piston term (total degree 0) is omitted.
    Fast evaluation is performed using NumPy arrays.
    """
    _monomial_cache = {}

    @classmethod
    def _get_monomial_indices(cls, maxdeg):
        """Return the (px, py) exponents for all monomials of total degree 1 to maxdeg.

        Uses a class-level cache to avoid recomputation for repeated calls with the same maxdeg.

        The returned array has shape (n_terms, 2), where each row gives the exponents (px, py)
        for the monomial x**px * y**py, ordered by increasing total degree and omitting the piston (degree 0).

        Parameters
        ----------
        maxdeg : int
            Maximum total degree of the monomials (excluding the piston term).

        Returns
        -------
        arr : ndarray of shape (n_terms, 2)
            Array of exponents for each monomial term.
        """
        if maxdeg not in cls._monomial_cache:
            # Number of monomials (excluding piston)
            n_terms = maxdeg * (maxdeg + 3) // 2
            arr = np.empty((n_terms, 2), dtype=int)
            idx = 0
            for d in range(1, maxdeg+1):
                for px in range(d, -1, -1):
                    py = d - px
                    arr[idx, 0] = px
                    arr[idx, 1] = py
                    idx += 1
            cls._monomial_cache[maxdeg] = arr
        return cls._monomial_cache[maxdeg]
    
    def __init__(self, A, R_norm=1.0):
        self.A = np.asarray(A, dtype=float, order="C")
        self.N = len(A) - 1
        self.R_norm = float(R_norm)
        # Compute the minimum degree that covers all N non-piston terms.
        # Number of monomials through degree d (excluding piston) = d*(d+3)//2.
        _maxdeg = 1
        while _maxdeg * (_maxdeg + 3) // 2 < self.N:
            _maxdeg += 1
        self._maxdeg = _maxdeg
        self._coef_array_xy = np.zeros((self._maxdeg + 1, self._maxdeg + 1), dtype=float, order="C")

        monomials = self._get_monomial_indices(self._maxdeg)
        n_terms = min(len(monomials), self.N)  # use exactly N terms
        pxs, pys = monomials[:n_terms, 0], monomials[:n_terms, 1]

        scales = np.power(self.R_norm, -(pxs + pys)) # Scale for normalized coordinates
        self._coef_array_xy[pxs, pys] += self.A[1:1+n_terms] * scales

        # Precompute gradient arrays
        self._coef_array_xy_gradx = np.zeros((self._maxdeg, self._maxdeg), dtype=float, order="C")
        self._coef_array_xy_grady = np.zeros((self._maxdeg, self._maxdeg), dtype=float, order="C")
        # For gradx: d/dx(x^p * y^q) = p * x^{p-1} * y^q
        px = np.arange(1, self._maxdeg + 1)[:, None]
        self._coef_array_xy_gradx[:, :] = px * self._coef_array_xy[1:, :-1]
        # For grady: d/dy(x^p * y^q) = q * x^p * y^{q-1}
        py = np.arange(1, self._maxdeg + 1)
        self._coef_array_xy_grady[:, :] = self._coef_array_xy[:-1, 1:] * py

    def eval(self, x, y):
        """
        Evaluate the polynomial surface at paired points (x, y).

        Computes sum_{p, q} coef[p, q] * x**p * y**q at each point (x[i], y[i]).

        Parameters
        ----------
        x, y : array_like
            Points at which to evaluate the surface. Must have the same shape.

        Returns
        -------
        Z : ndarray
            Surface value(s) at the given (x, y) points, same shape as x and y.
        """
        return np.polynomial.polynomial.polyval2d(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float),
            self._coef_array_xy
        )

    def gradx(self, x, y):
        """
        Evaluate the partial derivative with respect to x at paired points (x, y).

        Parameters
        ----------
        x, y : array_like
            Points at which to evaluate the x-derivative. Must have the same shape.

        Returns
        -------
        Zx : ndarray
            dz/dx value(s) at the given (x, y) points, same shape as x and y.
        """
        return np.polynomial.polynomial.polyval2d(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float),
            self._coef_array_xy_gradx
        )

    def grady(self, x, y):
        """
        Evaluate the partial derivative with respect to y at paired points (x, y).

        Parameters
        ----------
        x, y : array_like
            Points at which to evaluate the y-derivative. Must have the same shape.

        Returns
        -------
        Zy : ndarray
            dz/dy value(s) at the given (x, y) points, same shape as x and y.
        """
        return np.polynomial.polynomial.polyval2d(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float),
            self._coef_array_xy_grady
        )
    
class XPolynom(Surface):
    """Surface defined by a 2D polynomial expansion in $x$ and $y$ coordinates.

    The surface sag follows the equation:

    .. math::

        z(x, y) = \sum_{i=1}^N A_i m_i\left(\frac{x-x_0}{R_{\mathrm{norm}}}, \frac{y-y_0}{R_{\mathrm{norm}}}\right)

    where :math:`m_i(x, y) = x^{p_i} y^{q_i}` is the i-th individual monomial with exponents
    :math:`(p_i, q_i)` ordered by increasing total degree :math:`p_i + q_i` and, for equal
    total degree, by decreasing x-power (i.e., x, y, x^2, xy, y^2, x^3, ...),
    :math:`A_i` are the polynomial coefficients, :math:`R_{\mathrm{norm}}` is the normalization radius,
    and :math:`(x_0, y_0)` is the origin for the polynomial expansion.

    The piston term (total degree 0) is omitted.
    Fast evaluation and derivatives are performed using precomputed NumPy arrays.

    Parameters
    ----------
    coef : list of float
        Polynomial coefficients $A_i$ (excluding piston).
    R_norm : float, optional
        Normalization radius for $x$ and $y$ coordinates. Default is 1.0.
    x_origin, y_origin : float, optional
        Origin for the polynomial expansion. Default is (0, 0).
    """
    def __init__(self, coef, R_norm=1.0, x_origin=0.0, y_origin=0.0):
        # Store polynomial coefficients and parameters
        self.coef = np.array(coef, dtype=float, order="C")
        self.R_norm = float(R_norm)
        self.x_origin = float(x_origin)
        self.y_origin = float(y_origin)

        # Build the polynomial representation (prepend 0 for piston)
        self.P = _ExtendedPolynomials([0.0] + list(coef), R_norm=self.R_norm)

        # Precompute coefficient arrays for evaluation and gradients
        self._xycoef = self.P._coef_array_xy
        self._xycoef_gradx = self.P._coef_array_xy_gradx
        self._xycoef_grady = self.P._coef_array_xy_grady

        # Create the underlying C++ polynomial surface for fast evaluation
        self._surface = _batoid.CPPPolynomialSurface(
            self._xycoef.ctypes.data,
            self._xycoef_gradx.ctypes.data,
            self._xycoef_grady.ctypes.data,
            self.x_origin, self.y_origin,
            self._xycoef.shape[0],
            self._xycoef.shape[1]
        )

    def __hash__(self):
        return hash((
            "batoid.XPolynom",
            tuple(self.coef),
            self.R_norm,
            self.x_origin,
            self.y_origin,
        ))

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return (self.coef, self.R_norm, self.x_origin, self.y_origin)

    def __eq__(self, rhs):
        if not isinstance(rhs, XPolynom):
            return False
        return (np.array_equal(self.coef, rhs.coef) and
                self.R_norm == rhs.R_norm and
                self.x_origin == rhs.x_origin and
                self.y_origin == rhs.y_origin)

    def __repr__(self):
        out = f"XPolynom({self.coef!r}"
        if self.R_norm != 1.0:
            out += f", R_norm={self.R_norm}"
        if self.x_origin != 0.0:
            out += f", x_origin={self.x_origin}"
        if self.y_origin != 0.0:
            out += f", y_origin={self.y_origin}"
        out += ")"
        return out
