import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class DifferentiableLensing(torch.nn.Module):
    """
    The differentiable lensing module that performs all required traversals through the lensing equation.

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
    1. Mini-grid fetching of quads- If you ever need to scale beyond 200×200, you can store quads in a spatial index (like a uniform grid or k-d tree). That way you only query the quads near each square, instead of all quads. But with ~40k cells, bounding-box + loop pruning will already be fast enough.
    2. Vectorised pre-filter computing to have a g1 x g1 x g2 x g2 shaped mask of where to compute areas- must be divided into mini-grids if g1 and g2 are > 100
    3. Sutherland-Hodgman optimisations for square grids-
        a. if quad is above the upper horizontal - skip checking with the rest of the square grid cells
        b. if quad is to the right of the right vertical - skip checking all rows leading up to the square grid cell
        c. if quad is to the left of the left vertical - skip the present row of the square grid cells
    4. For simpler deflection fields, a neighbourhood of the source grid can be checked for intersection with the destination grid, as further than that there would be no interactions (e.g., SIS deflection angle)
    """
    def __init__(self, device, alpha, target_resolution, target_shape):
        super(DifferentiableLensing, self).__init__()
        self.device = device
        self.target_resolution = target_resolution
        self.target_shape = target_shape
        self.half_arcsec_bound = target_resolution * target_shape / 2.0  # half-size in arcsec

        pos_x = torch.linspace(-self.half_arcsec_bound, self.half_arcsec_bound, target_shape, device=device)
        pos_y = torch.linspace(-self.half_arcsec_bound, self.half_arcsec_bound, target_shape, device=device)
        theta_y, theta_x = torch.meshgrid(pos_y, pos_x, indexing='ij')  # y: rows, x: cols
        theta_y, theta_x = theta_y.unsqueeze(0), theta_x.unsqueeze(0)
        self.register_buffer('theta_y', theta_y)
        self.register_buffer('theta_x', theta_x)
        if alpha != None:
            self.set_alpha(alpha)

    def set_alpha(self, alpha):
        if len(alpha.shape) == 3:
            alpha = alpha.unsqueeze(0)
        alpha_x, alpha_y = torch.split(alpha, [1,1], dim=1)
        beta_x = self.theta_x - alpha_x
        beta_y = self.theta_y - alpha_y
        beta_x_norm = beta_x / self.half_arcsec_bound
        beta_y_norm = beta_y / self.half_arcsec_bound
        grid = torch.stack((beta_x_norm, beta_y_norm), dim=-1).squeeze(1)
        self.register_buffer('grid', grid)
        self.register_buffer('beta_y', beta_y)
        self.register_buffer('beta_x', beta_x)

    def make_grid(self, lower_bound, upper_bound, shape):
        x = torch.linspace(lower_bound, upper_bound, shape)
        y = torch.linspace(lower_bound, upper_bound, shape)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # y: rows, x: cols
        grid_y, grid_x = grid_y.unsqueeze(0), grid_x.unsqueeze(0)
        return grid_x, grid_y
    
    def polygon_area(self, poly_xy, sign=False):
        """Signed area via the shoelace formula. poly_xy: (N,2) tensor."""
        x = poly_xy[:, 0]; y = poly_xy[:, 1]
        x1 = torch.roll(x, -1); y1 = torch.roll(y, -1)
        if sign:
            return 0.5 * torch.sum(x * y1 - x1 * y)
        return 0.5 * torch.abs(torch.sum(x * y1 - x1 * y))
    
    def intersect(self, p1, p2, nx, ny, c):
        """Compute intersection of line segment p1->p2 with clip edge defined by (nx,ny,c)."""
        x1,y1 = p1; x2,y2 = p2
        d1 = nx*x1 + ny*y1 + c
        d2 = nx*x2 + ny*y2 + c
        t = d1 / (d1 - d2)
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))

    def clip_polygon_with_square(self, polygon, x_min, x_max, y_min, y_max):
        """Clip polygon against an axis-aligned square."""
        # Define clipping edges as [((normal, offset), inside test)]
        clip_edges = [
            ((1, 0), -x_min),   # x >= x_min
            ((-1, 0), x_max),   # x <= x_max
            ((0, 1), -y_min),   # y >= y_min
            ((0, -1), y_max),   # y <= y_max
        ]

        output_poly = polygon
        for (nx, ny), c in clip_edges:
            input_poly = output_poly
            output_poly = []
            if not input_poly: 
                break

            for i in range(len(input_poly)):
                s = input_poly[i]
                e = input_poly[(i+1) % len(input_poly)]

                s_inside = (nx*s[0] + ny*s[1] + c >= 0)
                e_inside = (nx*e[0] + ny*e[1] + c >= 0)

                if s_inside and e_inside:
                    output_poly.append(e)
                elif s_inside and not e_inside:
                    # leaving clip region
                    inter = self.intersect(s, e, nx, ny, c)
                    output_poly.append(inter)
                elif not s_inside and e_inside:
                    # entering clip region
                    inter = self.intersect(s, e, nx, ny, c)
                    output_poly.append(inter)
                    output_poly.append(e)
                # else both outside → add nothing
        return output_poly

    def square_grid_crop(self, square_grid_x, square_grid_y, non_square_grid_x, non_square_grid_y):
        sq_shape = square_grid_y.shape[0]-1
        nsq_shape = non_square_grid_y.shape[0]-1
        nsq_grid_frac_in_sq_grid = torch.zeros(sq_shape, sq_shape, nsq_shape, nsq_shape)
        for s_i in tqdm(range(sq_shape), desc='Iterating through axis-oriented grid rows'):
            for s_j in range(sq_shape):
                x_min, x_max = square_grid_x[s_i, s_j], square_grid_x[s_i, s_j+1]
                y_min, y_max = square_grid_y[s_i, s_j], square_grid_y[s_i+1, s_j]
                for ns_i in range(nsq_shape):
                    for ns_j in range(nsq_shape):
                        # original quad vertices
                        poly = [
                            (non_square_grid_x[ns_i, ns_j],     non_square_grid_y[ns_i, ns_j]),
                            (non_square_grid_x[ns_i+1, ns_j],   non_square_grid_y[ns_i+1, ns_j]),
                            (non_square_grid_x[ns_i+1, ns_j+1], non_square_grid_y[ns_i+1, ns_j+1]),
                            (non_square_grid_x[ns_i, ns_j+1],   non_square_grid_y[ns_i, ns_j+1]),
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
                            nsq_grid_frac_in_sq_grid[s_i, s_j, ns_i, ns_j] = area
                        # if (s_i,s_j) in [(16, 16)]:
                        #     print("quad corners:", quad)
                        #     print("clipped_poly:", clipped_poly)
                        #     print(square_grid_x[s_i, s_j], square_grid_x[s_i, s_j])
                        #     print(square_grid_x[s_i+1, s_j], square_grid_x[s_i+1, s_j])
                        #     print(square_grid_x[s_i+1, s_j+1], square_grid_x[s_i+1, s_j+1])
                        #     print(square_grid_x[s_i, s_j+1], square_grid_x[s_i, s_j+1])

        return nsq_grid_frac_in_sq_grid
    
    def build_sparse_mapping(self, T, As = 1, device=None):
        Sx, Sy, Ix, Iy = T.shape
        P = Ix * Iy
        Q = Sx * Sy
        # Flatten (sx,sy) -> q; (ix,iy) -> p
        # Get nonzeros
        nz = T.nonzero(as_tuple=False)  # [nnz, 4] with cols [sx, sy, ix, iy]
        sx, sy, ix, iy = nz[:,0], nz[:,1], nz[:,2], nz[:,3]
        q = sx * Sy + sy
        p = ix * Iy + iy
        vals = T[sx, sy, ix, iy]       # overlap areas
        # Divide by source-cell area to get average-intensity operator
        # If your source grid spacing is Δβ (in arcsec), As = (Δβ)^2
        # If you already stored T in **area units**, compute As once:
        # As = (beta_step_x * beta_step_y)
        # If uniform square cells:
        # As = (beta_step)**2
        vals = vals / As[ix, iy]

        indices = torch.stack([q, p], dim=0)   # [2, nnz]
        M = torch.sparse_coo_tensor(indices, vals, size=(Q, P), device=device).coalesce()
        return M, (Sx, Sy, Ix, Iy)

    def reconstruct_source_sparse(self, I_img, M, shapes):
        # I_img: (B, C, Ix, Iy)
        Sx, Sy, Ix, Iy = shapes
        B, C = I_img.shape[0], I_img.shape[1]

        P = Ix * Iy
        Q = Sx * Sy

        # Flatten image pixels to vectors
        I_flat = I_img.reshape(B*C, P).T        # (P, B*C)
        # Sparse mm: (Q, P) @ (P, B*C) -> (Q, B*C)
        S_flat = torch.sparse.mm(M, I_flat)  # differentiable w.r.t. I_flat
        S = S_flat.T.contiguous().view(B, C, Sx, Sy)
        return S
    
    def compute_As(self, square_grid_x, square_grid_y):
        # square_grid_*: (Sx+1, Sy+1) in source-plane units (arcsec or rad), monotonic
        dx = square_grid_x[:-1, 1:] - square_grid_x[:-1, :-1]  # (Sx, Sy)
        dy = square_grid_y[1:, :-1] - square_grid_y[:-1, :-1]  # (Sx, Sy)
        As_map = dx * dy
        # If uniform:
        if torch.allclose(As_map, As_map.mean()):
            return As_map.mean()         # scalar As
        else:
            return As_map                # per-cell area if non-uniform
    
    def forward(self, source_image):
        """
        source_image: (B, 1, H, W) source-plane image
        """
        grid = self.grid * torch.ones(source_image.shape[0], 1, 1, 1)
        # Sample source plane image
        lensed_image = F.grid_sample(source_image, grid, mode='nearest',
                                     padding_mode='zeros', align_corners=True)
        return lensed_image
    
    def backward_lensing(self, beta_x, beta_y, alpha):
        alpha_x, alpha_y = torch.split(alpha, [1,1], dim=0)
        theta_x, theta_y = beta_x + alpha_x[0], beta_y + alpha_y[0]
        return theta_x, theta_y

    def backward(self, lensing_image, kernel_pass=False):
        """
        lensing_image: (B, 1, H, W) source-plane image
        """
        B, c, x, y = lensing_image.shape
        beta_x, beta_y = self.beta_x * torch.ones(lensing_image.shape[0], 1, 1, 1), self.beta_y * torch.ones(lensing_image.shape[0], 1, 1, 1)
        pos_x, pos_y = (beta_x + self.half_arcsec_bound) / self.target_resolution, (beta_y + self.half_arcsec_bound) / self.target_resolution
        image_flux = lensing_image.view(B, c, x * y)
        px = pos_x.view(B, x * y).type(torch.int64)
        py = pos_y.view(B, x * y).type(torch.int64)
        px = torch.clamp(px, 0, x-1)
        py = torch.clamp(py, 0, y-1)

        flattened_indices = py * y + px

        reconstructed_source = torch.zeros(B, c, x * y, device=self.device)
        weight_map = torch.zeros(B, 1, x * y, device=self.device)

        for c_i in range(c):
            reconstructed_source[:, c_i].scatter_add_(1, flattened_indices, image_flux[:, c_i])
        weight_map[:, 0].scatter_add_(1, flattened_indices, torch.ones_like(px).float())

        div_by_zero_mask = weight_map > 0
        reconstructed_source[div_by_zero_mask.expand_as(reconstructed_source)] /= weight_map.expand_as(reconstructed_source)[div_by_zero_mask.expand_as(reconstructed_source)]

        reconstructed_source = reconstructed_source.view(B, c, x, y)
        weight_map = weight_map.view(B, 1, x, y)
        if not kernel_pass: reconstructed_source = self.convolve_gaussian(reconstructed_source)
        return reconstructed_source, weight_map

    def construct_sis(self, alpha_r):
        r = torch.sqrt(self.theta_x**2 + self.theta_y**2)
        alpha_x = alpha_r * self.theta_x / r
        alpha_y = alpha_r * self.theta_y / r
        # fix the origin singularity
        alpha_x = torch.nan_to_num(alpha_x, nan=0.0)
        alpha_y = torch.nan_to_num(alpha_y, nan=0.0)
        alpha = torch.cat([alpha_x, alpha_y], dim=0)
        return alpha
    
    # r = torch.sqrt(self.theta_x**2 + self.theta_y**2)
    #     alpha_r_vec = torch.ones_like(self.theta_x) * alpha_r
    #     # alpha_r_vec = torch.sqrt(alpha_x**2 + alpha_y**2)
    #     theta_r = torch.cat([self.theta_x, self.theta_y], dim=0)
    #     r_vecs = torch.linalg.norm(theta_r, dim=0, keepdim=True)
    #     alpha_r_close_to_origin_mask = alpha_r_vec > r_vecs
    #     alpha_r_vec[alpha_r_close_to_origin_mask] = r_vecs[alpha_r_close_to_origin_mask]
    #     alpha_x = alpha_r_vec * self.theta_x / r
    #     alpha_y = alpha_r_vec * self.theta_y / r
    #     # fix the origin singularity
    #     alpha_x = torch.nan_to_num(alpha_x, nan=0.0)
    #     alpha_y = torch.nan_to_num(alpha_y, nan=0.0)
    #     alpha = torch.cat([alpha_x, alpha_y], dim=0)
    #     return alpha
    
    def get_gaussian_kernel(self, sigma, size):
        """Returns a 2D Gaussian kernel with the specified size and sigma.
        
        Parameters
        ----------
        size : int, the side length of the square kernel (must be odd for centering)
        sigma : float, the standard deviation (sigma) of the Gaussian distribution
        
        Returns
        -------
        kernel : array, shape = (size, size)
            A 2D array representing the centered Gaussian kernel.
        """
        # Create a 1D array of positions centered at zero
        x = torch.linspace(-(size // 2), size // 2, size)
        # Scale the positions by sigma
        x /= np.sqrt(2) * sigma
        # Square the values
        x2 = x ** 2
        # Compute the 2D kernel using broadcasting
        kernel = np.exp(- x2[:, None] - x2[None, :])
        # Normalize the kernel so that the sum of all elements is 1
        return (kernel / kernel.sum()).unsqueeze(0)   