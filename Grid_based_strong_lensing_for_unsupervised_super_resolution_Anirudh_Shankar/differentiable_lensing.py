import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

EPS = 1e-7

def shared_colorimshow(images, nrows, ncols, cmap = None, colorbar_label = None, **kwargs):
    """
    Display a list/sequence of 2D images in a shared color scale.

    images: iterable of 2D numpy arrays or tensors convertible to numpy
    nrows, ncols: layout of subplots
    cmap: matplotlib colormap (default 'viridis')
    colorbar_label: optional label for the colorbar
    **kwargs: intended to allow calling axis methods like 'set_title' etc.
              NOTE: the function uses exec(...) on kwargs values.
    Returns: (plot, axes) tuple from matplotlib
    """
    if cmap == None: cmap = 'viridis'

    # find global min/max across all images to share the color scale
    vmin, vmax = np.inf, -np.inf
    for image in images:
        imin, imax = image.min(), image.max()
        if imin < vmin: vmin = imin
        if imax > vmax: vmax = imax

    # create the figure and axes grid
    plot, axes = plt.subplots(nrows, ncols,)
    plot.set_size_inches(5 * ncols, 5 * nrows)

    for k in range(len(images)):
        i = k // ncols  # row index
        j = k % ncols   # column index

        # Because plt.subplots returns axes shaped differently depending on nrows/ncols,
        # the code branches to index the axes correctly.
        if nrows == 1 and ncols == 1:
            im = axes.imshow(images[k], cmap=cmap, vmin=vmin, vmax=vmax)
            for key, value in kwargs.items():
                # WARNING: using exec on runtime strings is both unsafe and hard to debug.
                # It also assumes 'value' is an expression indexing the right image.
                exec('axes[j].%s(%s[k])'%(key, value))
        elif nrows == 1:
            im = axes[j].imshow(images[k], cmap=cmap, vmin=vmin, vmax=vmax)
            for key, value in kwargs.items():
                exec('axes[j].%s(%s[k])'%(key, value))
        elif ncols == 1:
            im = axes[i].imshow(images[k], cmap=cmap, vmin=vmin, vmax=vmax)
            for key, value in kwargs.items():
                exec('axes[i].%s(%s[k])'%(key, value))
        else:
            im = axes[i, j].imshow(images[k], cmap=cmap, vmin=vmin, vmax=vmax)
            for key, value in kwargs.items():
                exec('axes[i][j].%s(%s[k])'%(key, value))

    # create a colorbar on the right
    plot.subplots_adjust(right=0.8)
    cbar_ax = plot.add_axes([0.85, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    cbar = plot.colorbar(im, cax=cbar_ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    return plot, axes


class DifferentiableLensing(torch.nn.Module):
    """
    Differentiable lensing module used to transform (lens) images by a deflection field.

    High-level behaviour:
    - Builds a theta grid (image-plane coordinates) stored in buffers.
    - Supports creation of several grid types (regular center-grid and log grid).
    - Implements polygon clipping and area computation to compute how distorted source pixels
      overlap destination pixels (used to construct sparse linear mappings).
    - Provides forward/backward lensing helpers and an actual differentiable forward() that
      uses a pre-computed 'grid' with torch.nn.functional.grid_sample to resample a source image.
    
    Attributes
    ----------

    device: string
        Whether the tensors are stored in the cpu or the gpu (if present)
    alpha: torch.Tensor
        The deflection field the module can be initialised with
    target_resolution: float
        The arcsec per pixel resolution used in grid construction
    target_shape: uint
        The number of pixels to be used in grid construction

    Methods
    -------

    set_alpha(alpha)
        The deflection field is used to construct and store the source intensity positions
    make_grid(lower_bound, upper_bound, shape)
        Helper function used to create an axis oriented grid of a certain shape within the given bounds
    polygon_area(poly_xy, sign=False)
        The method that employs the shoelace formula to compute the signed area of a polygon
    intersect(p1, p2, nx, ny, c)
        Helper function to compute the intersection of line segment p1->p2 with the clip edge defined by (nx,ny,c)
    clip_polygon_with_square(polygon, x_min, x_max, y_min, y_max)
        Helper function to clip a polygon against an axis-aligned square
    square_grid_crop(square_grid_x, square_grid_y, non_square_grid_x, non_square_grid_y)
        Computes overlap area fractions between each cell of a square grid and each square of a distorted grid
    build_sparse_mapping(T, As = 1, device=None)
        Optimises the overlap area fraction cross-grid to a sparse coo matrix, normalised by pixel areas to conserve surface intensity
    reconstruct_source_sparse(I_img, M, shapes)
        Scatters the intensities from a given image to construct a new one as directed by the sparse coo matrix
    compute_As(square_grid_x, square_grid_y)
        Helper function to compute the cell-wise area of a grid
    forward(source_image)
        Forward pass through the lensing equation to lens a source image using grid_sample
    backward_lensing(beta_x, beta_y, alpha)
        Backward pass through the lensing equation to compute betas from thetas
    construct_sis(alpha_r)
        Helper function to construct a single-isothermal-sphere lens' deflection field

    TODO: 
    Cross-grid calculation
    1. Mini-grid fetching of quads- If you ever need to scale beyond 200×200, you can store quads in a spatial index (like a uniform grid or k-d tree). That way you only query the quads near each square, instead of all quads. But with ~40k cells, bounding-box + loop pruning will already be fast enough.
    2. Vectorised pre-filter computing to have a g1 x g1 x g2 x g2 shaped mask of where to compute areas- must be divided into mini-grids if g1 and g2 are > 100
    3. Sutherland-Hodgman optimisations for square grids-
        a. if quad is above the upper horizontal - skip checking with the rest of the square grid cells
        b. if quad is to the right of the right vertical - skip checking all rows leading up to the square grid cell
        c. if quad is to the left of the left vertical - skip the present row of the square grid cells
    4. For simpler deflection fields, a neighbourhood of the source grid can be checked for intersection with the destination grid, as further than that there would be no interactions (e.g., SIS deflection angle)
    
    Module structure
    1. Train/test should not need to use the lensing module, i.e., all related functionalities to be declared as functions outside the module, e.g., PSF
    
    Misc:
    - Some helper routines convert tensors to Python floats (non-differentiable, CPU).
    - Many loops are pure-Python / inefficient for large grids.
    
    
    """
    def __init__(self, device, alpha, target_resolution, target_shape):
        super(DifferentiableLensing, self).__init__()
        self.device = device
        self.target_resolution = target_resolution
        self.target_shape = target_shape
        self.half_arcsec_bound = target_resolution * target_shape / 2.0  # half-size in arcsec

        # create axis-oriented coordinates for the image plane (theta)
        pos_x = torch.linspace(-self.half_arcsec_bound, self.half_arcsec_bound, target_shape, device=device)
        pos_y = torch.linspace(-self.half_arcsec_bound, self.half_arcsec_bound, target_shape, device=device)
        # meshgrid with indexing='ij' gives (rows=y, cols=x)
        theta_y, theta_x = torch.meshgrid(pos_y, pos_x, indexing='ij')  # y: rows, x: cols
        theta_y, theta_x = theta_y.unsqueeze(0), theta_x.unsqueeze(0)  # add channel-like dim for convenience
        # store as buffers so they are moved with .to(device) and saved with state_dict, but not parameters
        self.register_buffer('theta_y', theta_y)
        self.register_buffer('theta_x', theta_x)
        if alpha != None:
            self.set_alpha(alpha)
    
    def make_center_grid(self, lower_bound, upper_bound, shape):
        """
        Returns pixel-center coordinates for a square grid.

        Returns:
          grid_x, grid_y: center coordinates shaped (shape, shape) using indexing='xy'
                         (grid_x[i,j] = x coordinate for column j, grid_y[i,j] = y coordinate for row i)
          odd_grid_x, odd_grid_y: the (shape+1, shape+1) coordinates of pixel edges (useful for polygon vertices)
        Side effect:
          sets self.pixel_width (scalar) = pixel_width in same units as bounds
        """
        pixel_width = (upper_bound - lower_bound) / shape
        x_centers = torch.linspace(lower_bound + pixel_width/2,
                                   upper_bound - pixel_width/2,
                                   shape, device=self.theta_x.device)
        y_centers = torch.linspace(lower_bound + pixel_width/2,
                                   upper_bound - pixel_width/2,
                                   shape, device=self.theta_x.device)

        # indexing='xy' gives shapes (len(y_centers), len(x_centers)) consistent with the docstring
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing='xy')

        # odd grid (pixel edges): shape+1 points along each axis
        odd_x = torch.linspace(lower_bound,
                               upper_bound,
                               shape+1, device=self.theta_x.device)
        odd_y = torch.linspace(lower_bound,
                               upper_bound,
                               shape+1, device=self.theta_x.device)
        odd_grid_x, odd_grid_y = torch.meshgrid(odd_x, odd_y, indexing='xy')

        self.pixel_width = pixel_width
        return grid_x, grid_y, odd_grid_x, odd_grid_y

    
    def make_log_grid(self, lower_bound, upper_bound, shape, c):
        """
        Make a non-uniform 'log-like' grid concentrated around center.
        c controls the stretching.

        Returns:
          grid_x, grid_y (centers),
          odd_grid_x, odd_grid_y (edges) — note these are built from a transformed
          monotonic set of grid points and returned as torch tensors.
        Sets self.pixel_width (still computed as uniform spacing of the original bounds).
        """
        pixel_width = (upper_bound - lower_bound) / shape
        x_centers = torch.linspace(lower_bound + pixel_width/2,
                                   upper_bound - pixel_width/2,
                                   shape)
        y_centers = torch.linspace(lower_bound + pixel_width/2,
                                   upper_bound - pixel_width/2,
                                   shape)
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing='ij')

        # Build a specially spaced vector: first_half_log and second_half_log, combined and scaled by c.
        # Watch out: this uses torch.logspace with `base=torch.e` and some +1/-1 shifts to avoid singularities.
        first_half_log = (-torch.logspace(0, np.log(c*(upper_bound)+1), shape // 2 + 1, base=torch.e).flip(dims=[-1]))+1
        torch.nan_to_num_(first_half_log, nan=0.0, posinf=0.0, neginf=0.0)
        second_half_log = torch.logspace(0, np.log(c*(upper_bound)+1), shape // 2 + 1, base=torch.e)[1:]-1
        torch.nan_to_num_(second_half_log, nan=0.0, posinf=0.0, neginf=0.0)
        grid_points = torch.cat([first_half_log, second_half_log])
        grid_points = grid_points / c

        # odd grids built from grid_points (edges)
        odd_grid_y, odd_grid_x = torch.meshgrid(grid_points, grid_points, indexing='ij')

        self.pixel_width = pixel_width
        return grid_x, grid_y, odd_grid_x, odd_grid_y

    def polygon_area(self, poly_xy, sign=False):
        """Signed or unsigned polygon area using the shoelace formula.
        poly_xy: (N,2) tensor of vertices in order.
        Returns 0.5 * abs(sum(x_i*y_{i+1} - x_{i+1}*y_i))
        If sign=True, returns signed area (can be negative depending on winding).
        """
        x = poly_xy[:, 0]; y = poly_xy[:, 1]
        x1 = torch.roll(x, -1); y1 = torch.roll(y, -1)
        if sign:
            return 0.5 * torch.sum(x * y1 - x1 * y)
        return 0.5 * torch.abs(torch.sum(x * y1 - x1 * y))
    
    def intersect(self, p1, p2, nx, ny, c):
        """
        Compute intersection of segment p1->p2 with the line (nx * x + ny * y + c = 0).

        Important:
        - The implementation converts coordinates to Python floats via float(...).
          That means you lose autograd and device info here — it's not differentiable.
        - The function clamps t to [0,1], returning the (possibly clamped) intersection point.
        - If denominator is (near) zero (segment parallel to edge), it uses t=0.5 midpoint,
          which is a heuristic not strictly geometrically correct but avoids NaNs.
        """
        x1,y1 = float(p1[0]), float(p1[1])
        x2,y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        denom = nx*dx + ny*dy
        numer = -(nx*x1 + ny*y1 + c)
        if abs(denom) < 1e-12:
            t = 0.5
        else:
            t = numer / denom
            # clamp to segment
            t = max(0.0, min(1.0, t))
        xi = x1 + t*dx
        yi = y1 + t*dy
        return (xi, yi)


    def clip_polygon_with_square(self, polygon, x_min, x_max, y_min, y_max):
        """
        Clip a (possibly non-convex) polygon against an axis-aligned square
        using the Sutherland–Hodgman clipping algorithm.

        polygon: iterable of (x,y) points (e.g., list of tuples)
        returns: output_poly as list of (x,y) points inside the square
        """
        def is_inside(pt, nx, ny, c, eps=EPS):
            x,y = pt
            return (nx*x + ny*y + c) >= -eps
        
        # The clip_edges are expressed as (normal_x, normal_y), c such that inside satisfies
        # nx * x + ny * y + c >= 0
        clip_edges = [
            ((1, 0), -x_min),   # x >= x_min  => 1*x + 0*y - x_min >= 0
            ((-1, 0), x_max),   # x <= x_max  => -1*x + 0*y + x_max >= 0
            ((0, 1), -y_min),   # y >= y_min
            ((0, -1), y_max),   # y <= y_max
        ]

        output_poly = polygon
        for (nx, ny), c in clip_edges:
            input_poly = output_poly
            output_poly = []
            if len(input_poly) == 0:
                break

            for i in range(len(input_poly)):
                s = input_poly[i]
                e = input_poly[(i+1) % len(input_poly)]

                s_inside = is_inside(s, nx, ny, c)
                e_inside = is_inside(e, nx, ny, c)

                if s_inside and e_inside:
                    # both inside: keep the end vertex
                    output_poly.append(e)
                elif s_inside and not e_inside:
                    # leaving clipping region: append intersection point
                    inter = self.intersect(s, e, nx, ny, c)
                    output_poly.append(inter)
                elif not s_inside and e_inside:
                    # entering clipping region: append intersection then the end vertex
                    inter = self.intersect(s, e, nx, ny, c)
                    output_poly.append(inter)
                    output_poly.append(e)
                # both outside: nothing to add
        return output_poly

    def square_grid_crop(self, square_grid_x, square_grid_y, non_square_grid_x, non_square_grid_y):
        """
        For each cell of an axis-aligned square grid (square_grid), compute the overlap area
        with each quad in a non-square (distorted) grid (non_square_grid). Returns a tensor
        with shape (Sx, Sy, nsq_shape-1, nsq_shape-1) containing areas.

        Uses explicit python loops + tqdm, so this is slow for large grids (O(S^2 * nsq^2)).
        """
        sq_shape = square_grid_y.shape[0]
        nsq_shape = non_square_grid_y.shape[0]
        nsq_grid_frac_in_sq_grid = torch.zeros(sq_shape, sq_shape, nsq_shape-1, nsq_shape-1)

        for s_i in tqdm(range(sq_shape), desc='Iterating through axis-oriented grid rows'):
            for s_j in range(sq_shape):
                x_c, y_c = square_grid_x[s_i,s_j], square_grid_y[s_i,s_j]
                x_min, x_max = x_c - self.pixel_width/2, x_c + self.pixel_width/2
                y_min, y_max = y_c - self.pixel_width/2, y_c + self.pixel_width/2

                for ns_i in range(0, nsq_shape-1):
                    for ns_j in range(0, nsq_shape-1):
                        # vertices of the non-square quad, ordered to form a polygon
                        poly = [
                            (non_square_grid_x[ns_i+1, ns_j], non_square_grid_y[ns_i+1, ns_j]),
                            (non_square_grid_x[ns_i+1, ns_j+1], non_square_grid_y[ns_i+1, ns_j+1]),
                            (non_square_grid_x[ns_i, ns_j+1], non_square_grid_y[ns_i, ns_j+1]),
                            (non_square_grid_x[ns_i, ns_j], non_square_grid_y[ns_i, ns_j]),
                        ]
                        quad = torch.tensor(poly)
                        quad_x_min = quad[:, 0].min()
                        quad_x_max = quad[:, 0].max()
                        quad_y_min = quad[:, 1].min()
                        quad_y_max = quad[:, 1].max()

                        # quick bounding-box rejection
                        if quad_x_max < x_min or quad_x_min > x_max or quad_y_max < y_min or quad_y_min > y_max:
                            continue

                        clipped_poly = self.clip_polygon_with_square(poly, x_min, x_max, y_min, y_max)
                        if clipped_poly:
                            clipped = clipped_poly
                            if len(clipped) < 3:
                                continue
                            poly_t = torch.tensor(clipped, dtype=torch.float32)
                            area = self.polygon_area(poly_t)
                            nsq_grid_frac_in_sq_grid[s_i, s_j, ns_i, ns_j] = area

        return nsq_grid_frac_in_sq_grid
    
    def log_grid_crop(self, log_grid_x, log_grid_y, non_square_grid_x, non_square_grid_y):
        """
        Same idea as square_grid_crop but for a log-spaced grid whose cells are defined by
        adjacent grid points in log_grid_x/log_grid_y (note: log_grid shapes are likely different).
        Returns per-cell overlap areas shaped (log_shape-1, log_shape-1, nsq-1, nsq-1).
        """
        log_shape = log_grid_y.shape[0]
        nsq_shape = non_square_grid_y.shape[0]
        nsq_grid_frac_in_sq_grid = torch.zeros(log_shape-1, log_shape-1, nsq_shape-1, nsq_shape-1)
        for l_i in tqdm(range(log_shape-1), desc='Iterating through axis-oriented grid rows'):
            for l_j in range(log_shape-1):
                x_min, x_max = log_grid_x[l_i, l_j], log_grid_x[l_i, l_j+1]
                y_min, y_max = log_grid_y[l_i, l_j], log_grid_y[l_i+1, l_j]
                for ns_i in range(0, nsq_shape-1):
                    for ns_j in range(0, nsq_shape-1):
                        poly = [
                            (non_square_grid_x[ns_i+1, ns_j], non_square_grid_y[ns_i+1, ns_j]),
                            (non_square_grid_x[ns_i+1, ns_j+1], non_square_grid_y[ns_i+1, ns_j+1]),
                            (non_square_grid_x[ns_i, ns_j+1], non_square_grid_y[ns_i, ns_j+1]),
                            (non_square_grid_x[ns_i, ns_j], non_square_grid_y[ns_i, ns_j]),
                        ]
                        quad = torch.tensor(poly)
                        quad_x_min = quad[:, 0].min()
                        quad_x_max = quad[:, 0].max()
                        quad_y_min = quad[:, 1].min()
                        quad_y_max = quad[:, 1].max()
                        if quad_x_max < x_min or quad_x_min > x_max or quad_y_max < y_min or quad_y_min > y_max:
                            continue
                        clipped_poly = self.clip_polygon_with_square(poly, x_min, x_max, y_min, y_max)
                        if clipped_poly:
                            area = self.polygon_area(torch.tensor(clipped_poly, dtype=torch.float32))
                            nsq_grid_frac_in_sq_grid[l_i, l_j, ns_i, ns_j] = area
        return nsq_grid_frac_in_sq_grid
    
    def build_sparse_mapping(self, T, As = 1, device=None):
        """
        Build a sparse COO matrix M mapping image-plane pixels (P) to source-plane pixels (Q)
        using the 4D overlap tensor T with shape (Sx, Sy, Ix, Iy).

        T[sx, sy, ix, iy] = overlap area of source cell (sx,sy) with image pixel (ix,iy).

        Returns:
          M: torch.sparse_coo_tensor of shape (Q, P) where Q = Sx*Sy, P = Ix*Iy
          shapes: (Sx, Sy, Ix, Iy) tuple to record the sizes
        """
        Sx, Sy, Ix, Iy = T.shape
        P = Ix * Iy
        Q = Sx * Sy

        # get non-zero indices in T: shape [nnz, 4]
        nz = T.nonzero(as_tuple=False)  # [nnz, 4] with cols [sx, sy, ix, iy]
        sx, sy, ix, iy = nz[:,0], nz[:,1], nz[:,2], nz[:,3]
        q = sx * Sy + sy  # flatten source index
        p = ix * Iy + iy  # flatten image index
        vals = T[sx, sy, ix, iy]       # overlap areas

        # Divide by source-cell area As to produce an operator that maps source intensities
        # to average intensity per image pixel. If As is scalar float, broadcast it.
        if type(As) == float:
            vals = vals / As
        else:
            vals = vals / As[ix, iy]

        indices = torch.stack([q, p], dim=0)   # [2, nnz]
        M = torch.sparse_coo_tensor(indices, vals, size=(Q, P), device=device).coalesce()
        return M, (Sx, Sy, Ix, Iy)


    def cross_grid_fill(self, I_img, Ms, shapes=None):
        """
        Apply a sequence (list) of sparse mappings `Ms` to an image I_img.
        I_img: (B, C, Ix, Iy)
        Ms: list of sparse (Q, P) matrices to apply sequentially

        Returns:
          S: (B, C, Sx, Sy) mapped image after applying all Ms
        """
        if shapes != None:
            Sx, Sy, Ix, Iy = shapes
        else:
            # fallback: assume square shape at last dim — fragile
            Sx, Sy, Ix, Iy = I_img.shape[-1], I_img.shape[-1], I_img.shape[-1], I_img.shape[-1]
        B, C = I_img.shape[0], I_img.shape[1]

        P = Ix * Iy
        Q = Sx * Sy

        # Flatten image pixels to vectors:
        # I_flat: (P, B*C)  -- column-major style so we can sparse-mm with M: (Q,P) @ (P, B*C)
        I_flat = I_img.reshape(B*C, P).T        # (P, B*C)
        for M in Ms:
            I_flat = torch.sparse.mm(M, I_flat)  # sparse mm: produces (Q, B*C)
        S = I_flat.T.contiguous().view(B, C, Sx, Sy)
        return S
    
    def nsq_As(self, nsq_grid_x, nsq_grid_y):
        """
        Compute cell areas for a non-square grid (nsq_grid_x/y are edges with shape (N+1,N+1)).

        Returns tensor of shape (N, N) with polygon areas for each cell.
        """
        nsq_shape = nsq_grid_x.shape[-1]
        areas = torch.zeros(nsq_shape-1,nsq_shape-1)
        for i in range(nsq_shape-1):
            for j in range(nsq_shape-1):
                polygon = [
                    (nsq_grid_x[i,j], nsq_grid_y[i,j]),
                    (nsq_grid_x[i+1,j], nsq_grid_y[i+1,j]),
                    (nsq_grid_x[i+1,j+1], nsq_grid_y[i+1,j+1]),
                    (nsq_grid_x[i,j+1], nsq_grid_y[i,j+1])
                ]
                areas[i,j] = self.polygon_area(torch.tensor(polygon, dtype=torch.float32))
        return areas
    
    def forward(self, source_image):
        """
        source_image: (B, C, H, W) source-plane image
        """
        grid = self.grid * torch.ones(source_image.shape[0], 1, 1, 1)
        # Sample source plane image
        lensed_image = F.grid_sample(source_image, grid, mode='nearest',
                                     padding_mode='zeros', align_corners=True)
        return lensed_image
    
    def backward_lensing(self, beta_x, beta_y, alpha):
        """
        Given beta (source-plane coords) and alpha (deflection), compute theta = beta + alpha
        alpha: expected shape [2, ...] or similar where splitting returns components.
        This method splits alpha by first dimension into (alpha_x, alpha_y).
        Note it indexes alpha_x[0] — ensure alpha's first dim matches expectations.
        """
        alpha_x, alpha_y = torch.split(alpha, [1,1], dim=0)
        theta_x, theta_y = beta_x + alpha_x[0], beta_y + alpha_y[0]
        return theta_x, theta_y
    
    def forward_lensing(self, theta_x, theta_y, alpha):
        """
        Computes beta = theta - alpha (forward lens equation).
        """
        alpha_x, alpha_y = torch.split(alpha, [1,1], dim=0)
        beta_x, beta_y = theta_x - alpha_x[0], theta_y - alpha_y[0]
        return beta_x, beta_y

    def construct_sis(self, theta_x, theta_y, alpha_r):
        """
        Build the deflection field (alpha_x, alpha_y) for a Single Isothermal Sphere (SIS) lens:
          alpha = alpha_r * (theta / r)

        Where alpha_r is a scalar (deflection amplitude).
        This handles the r=0 singularity by using torch.nan_to_num.
        Returns alpha as a concatenation of [alpha_x, alpha_y] along dim=0.
        """
        r = torch.sqrt(theta_x**2 + theta_y**2)
        r_shape = r.shape[-1]
        # avoid singular dividing by zero at center by forcing center to zero radius
        if r_shape % 2 == 1: r[0,r_shape//2, r_shape//2] = 0
        alpha_x = alpha_r * theta_x / r
        alpha_y = alpha_r * theta_y / r
        # remove NaNs/infs from division at r=0
        alpha_x = torch.nan_to_num(alpha_x, nan=0.0, posinf=0.0, neginf=0.0)
        alpha_y = torch.nan_to_num(alpha_y, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = torch.cat([alpha_x, alpha_y], dim=0)
        return alpha
    
    def gaussian_kernel(self, fwhm_arcsec:float, pixscale_arcsec:float, angle:float=0):
        """
        Create a Gaussian kernel in numpy arrays. Returns (Z, X, Y)
        Z: kernel values
        X, Y: integer coordinate grids (but note these are ints in the original code)
        """
        sigma_pix = (fwhm_arcsec / 2.3548) / (pixscale_arcsec)
        kernel_size = int(10 * sigma_pix)

        # The code creates integer arrays for coordinates which is odd — coordinates
        # and the kernel should be float arrays. Also kernel_size of zero possible -> handle.
        Z = np.zeros((2*kernel_size + 1, 2*kernel_size + 1), dtype=int)
        X = np.zeros(np.shape(Z), dtype=int)
        Y = np.zeros(np.shape(Z), dtype=int)

        for i in range(len(X)): # TODO: VECTORISE THIS
            for j in range(len(X)):
                X[i][j] = i - 2*kernel_size//2
                Y[i][j] = j - 2*kernel_size//2

        # rotate coordinate grid by -angle degrees (clockwise)
        angle_rad = np.radians(-angle)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        xy = np.vstack((X.flatten(), Y.flatten()))
        rotated_xy = np.dot(rotation_matrix, xy)
        rotated_x  = rotated_xy[0]   ;   rotated_y = rotated_xy[1]

        # gaussian formula
        Z = (1/(2 * np.pi * sigma_pix * sigma_pix)) * np.exp(-0.5 * (((rotated_x) / sigma_pix)**2 + ((rotated_y) / sigma_pix)**2))
        Z = Z.reshape(np.shape(X))
        return Z, X, Y

    def compute_variation_density(self, images):
        """
        Compute (sum of absolute) finite-difference variation across the batch.
        This is something like total variation but computed as:
          sum(|I[:, :, i] - I[:, :, i+1]|) / ((x_shape-1) * y_shape) +
          sum(|I[:, :, :, j] - I[:, :, :, j+1]|) / (x_shape * (y_shape-1))

        images expected shape: (B, C, x, y)
        Returns a scalar (sum of variations normalized by number of edges).
        """
        _, __, x_shape, y_shape = images.shape
        dx = torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :])
        dy = torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:])
        return torch.sum(dx)/((x_shape-1)*(y_shape)) + torch.sum(dy)/((x_shape)*(y_shape-1))
