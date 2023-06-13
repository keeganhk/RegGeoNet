from .pkgs import *

from .CD.chamferdist.chamfer import knn_points as knn_gpu
from .EMD.emd import earth_mover_distance_unwrapped
from .KNN_CPU import nearest_neighbors as knn_cpu
from .GS_CPU.cpp_subsampling import grid_subsampling as cpp_grid_subsample



def knn_on_gpu(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == 'cuda'
    assert query_pts.device.type == 'cuda'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_gpu(p1=query_pts, p2=source_pts, K=k, return_nn=False, return_sorted=True)[1]
    return knn_idx


def knn_on_cpu(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == 'cpu'
    assert query_pts.device.type == 'cpu'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_cpu.knn_batch(source_pts, query_pts, k, omp=True)
    return knn_idx


def knn_search(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == query_pts.device.type
    device_type = source_pts.device.type
    assert device_type in ['cpu', 'cuda']
    if device_type == 'cuda':
        knn_idx = knn_on_gpu(source_pts, query_pts, k)
    if device_type == 'cpu':
        knn_idx = knn_on_cpu(source_pts, query_pts, k)
    return knn_idx


def chamfer_distance_cuda(pts_s, pts_t, cpt_mode='max', return_detail=False):
    # pts_s: [B, Ns, C], source point cloud
    # pts_t: [B, Nt, C], target point cloud
    Bs, Ns, Cs, device_s = pts_s.size(0), pts_s.size(1), pts_s.size(2), pts_s.device
    Bt, Nt, Ct, device_t = pts_t.size(0), pts_t.size(1), pts_t.size(2), pts_t.device
    assert Bs == Bt
    assert Cs == Ct
    assert device_s == device_t
    assert device_s.type == 'cuda' and device_t.type == 'cuda'
    assert cpt_mode in ['max', 'avg']
    lengths_s = torch.ones(Bs, dtype=torch.long, device=device_s) * Ns
    lengths_t = torch.ones(Bt, dtype=torch.long, device=device_t) * Nt
    source_nn = knn_gpu(pts_s, pts_t, lengths_s, lengths_t, 1)
    target_nn = knn_gpu(pts_t, pts_s, lengths_t, lengths_s, 1)
    source_dist, source_idx = source_nn.dists.squeeze(-1), source_nn.idx.squeeze(-1) # [B, Ns]
    target_dist, target_idx = target_nn.dists.squeeze(-1), target_nn.idx.squeeze(-1) # [B, Nt]
    batch_dist = torch.cat((source_dist.mean(dim=-1, keepdim=True), target_dist.mean(dim=-1, keepdim=True)), dim=-1) # [B, 2]
    if cpt_mode == 'max':
        cd = batch_dist.max(dim=-1)[0].mean()
    if cpt_mode == 'avg':
        cd = batch_dist.mean(dim=-1).mean()
    if not return_detail:
        return cd
    else:
        return cd, source_dist, source_idx, target_dist, target_idx


def earth_mover_distance_cuda(pts_1, pts_2):
    # pts_1: [B, N1, C=1,2,3]
    # pts_2: [B, N2, C=1,2,3]
    assert pts_1.size(0) == pts_2.size(0)
    assert pts_1.size(2) == pts_2.size(2)
    assert pts_1.device == pts_2.device
    B, N1, C, device = pts_1.size(0), pts_1.size(1), pts_1.size(2), pts_1.device
    B, N2, C, device = pts_2.size(0), pts_2.size(1), pts_2.size(2), pts_2.device
    assert device.type == 'cuda'
    assert C in [1, 2, 3]
    if C < 3:
        pts_1 = torch.cat((pts_1, torch.zeros(B, N1, 3-C).to(device)), dim=-1) # [B, N1, 3]
        pts_2 = torch.cat((pts_2, torch.zeros(B, N2, 3-C).to(device)), dim=-1) # [B, N2, 3]  
    dist_1 = earth_mover_distance_unwrapped(pts_1, pts_2, transpose=False) / N1 # [B]
    dist_2 = earth_mover_distance_unwrapped(pts_2, pts_1, transpose=False) / N2 # [B]
    emd = ((dist_1 + dist_2) / 2).mean()
    return emd


def grid_subsample_cpu(pts, grid_size):
    # pts: [num_input, 3]
    # the output "pts_gs" is not the subset of the input "pts"
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.device.type == 'cpu'
    pts_gs = cpp_grid_subsample.compute(pts, sampleDl=grid_size) # (num_gs, 3)
    return pts_gs


def is_square_number(n):
    assert n == (int(np.sqrt(n)) ** 2), 'not a square number.'


def align_number(raw_number, expected_num_digits):
    # align a number string
    string_number = str(raw_number)
    ori_num_digits = len(string_number)
    assert ori_num_digits <= expected_num_digits
    return (expected_num_digits - ori_num_digits) * '0' + string_number


def load_pc(load_path):
    pcd = o3d.io.read_point_cloud(load_path)
    assert pcd.has_points()
    points = np.asarray(pcd.points) # (num_points, 3)
    attributes = {'colors': None, 'normals': None}
    if pcd.has_colors():
        colors = np.asarray(pcd.colors) # (num_points, 3)
        attributes['colors'] = colors
    if pcd.has_normals():
        normals = np.asarray(pcd.normals) # (num_points, 3)
        attributes['normals'] = normals
    return points, attributes


def save_pc(save_path, points, colors=None, normals=None):
    assert save_path[-3:] == 'ply', 'not .ply file'
    if type(points) == torch.Tensor:
        points = np.asarray(points.detach().cpu())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 
    if colors is not None:
        if type(colors) == torch.Tensor:
            colors = np.asarray(colors.detach().cpu())
        assert colors.min()>=0 and colors.max()<=1
        pcd.colors = o3d.utility.Vector3dVector(colors) # should be within the range of [0, 1]
    if normals is not None:
        if type(normals) == torch.Tensor:
            normals = np.asarray(normals.detach().cpu())
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True) # should be saved as .ply file


def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_mmn = (x - x_min) / (x_max - x_min)
    return x_mmn


def show_image(img):
    # input type: torch.tensor or numpy.ndarray
    # input size: 3xHxW or 1xHxW or HxW
    assert img.ndim in [2, 3]
    if img.ndim == 3:
        assert img.shape[0] in [1, 3]
    img = (min_max_normalization(np.asarray(img)) * 255.0).astype(np.uint8)
    if img.ndim==2 or img.shape[0]==1:
        img = img.squeeze() # HxW
        img_pil = Image.fromarray(img).convert('L')
    else:
        img_pil = Image.fromarray(np.transpose(img, (1, 2, 0)))
    return img_pil


def build_colormap(num_colors):
    # cmap: (num_colors, 3), within the range of [0, 1]
    queries = np.linspace(0, 1, num_colors, dtype=np.float32)
    cmap = matplotlib.cm.jet(queries)[:, 0:3] # (num_colors, 3)
    return cmap


def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    np.random.seed(worker_info.seed % 2**32)


def fps(xyz, num_sample):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = torch.randint(0, num_input, (batch_size,), dtype=torch.long).to(device)
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def fps_fix_first(xyz, num_sample):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = (xyz**2).sum(dim=-1).argmin(dim=-1).long().to(device)
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def index_points(pc, idx):
    # pc: [B, N, C]
    # 1) idx: [B, S] -> pc_selected: [B, S, C]
    # 2) idx: [B, S, K] -> pc_selected: [B, S, K, C]
    device = pc.device
    B = pc.shape[0]
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    pc_selected = pc[batch_indices, idx, :]
    return pc_selected


def random_sampling(pc, num_sample):
    # pc: (num_points, num_channels)
    # pc_sampled: # [num_sample, num_channels]
    num_points, num_channels = pc.shape
    assert num_sample < num_points
    selected_indices = np.random.choice(num_points, num_sample, replace=False) # (num_sample,)
    pc_sampled = pc[selected_indices, :]
    return pc_sampled


def farthest_point_sampling(pc, num_sample):
    # pc: (num_points, num_channels)
    # pc_sampled: [num_sample, num_channels]
    num_points, num_channels = pc.shape
    assert num_sample < num_points
    xyz = pc[:, 0:3] # sampling is based on spatial distance
    centroids = np.zeros((num_sample,))
    distance = np.ones((num_points,)) * 1e10
    farthest = np.random.randint(0, num_points)
    for i in range(num_sample):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pc_sampled = pc[centroids.astype(np.int32)]
    return pc_sampled


def farthest_point_sampling_fix_first(pc, num_sample):
    # pc: (num_points, num_channels)
    # pc_sampled: [num_sample, num_channels]
    num_points, num_channels = pc.shape
    assert num_sample < num_points
    xyz = pc[:, 0:3] # sampling is based on spatial distance
    centroids = np.zeros((num_sample,))
    distance = np.ones((num_points,)) * 1e10
    farthest = (xyz**2).sum(axis=-1).argmin()
    for i in range(num_sample):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pc_sampled = pc[centroids.astype(np.int32)]
    return pc_sampled


def axis_rotation(pc, angle, axis):
    # pc: (num_points, num_channels=3/6)
    # angle: [0, 2*pi]
    # axis: 'x', 'y', 'z'
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    assert angle>=0 and angle <= (2*np.pi)
    assert axis in ['x', 'y', 'z']
    # generate the rotation matrix
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    if axis == 'x':
        rot_mat = np.array([ [1, 0, 0], [0, c, -s], [0, s, c] ]).astype(np.float32)
    if axis == 'y':
        rot_mat = np.array([ [c, 0, s], [0, 1, 0], [-s, 0, c] ]).astype(np.float32)
    if axis == 'z':
        rot_mat = np.array([ [c, -s, 0], [s, c, 0], [0, 0, 1] ]).astype(np.float32)
    # apply the rotation matrix
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    return pc_rotated


def random_axis_rotation(pc, axis, return_angle=False):
    # pc: (num_points, num_channels=3/6)
    # axis: 'x', 'y', 'z'
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    assert axis in ['x', 'y', 'z']
    # generate a random rotation matrix
    angle = np.random.uniform() * 2 * np.pi
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    if axis == 'x':
        rot_mat = np.array([ [1, 0, 0], [0, c, -s], [0, s, c] ]).astype(np.float32)
    if axis == 'y':
        rot_mat = np.array([ [c, 0, s], [0, 1, 0], [-s, 0, c] ]).astype(np.float32)
    if axis == 'z':
        rot_mat = np.array([ [c, -s, 0], [s, c, 0], [0, 0, 1] ]).astype(np.float32)
    # apply the rotation matrix
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    if not return_angle:
        return pc_rotated
    else:
        return pc_rotated, angle
    
    
def random_rotation(pc, return_angle=False):
    # pc: (num_points, num_channels=3/6)
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    rot_mat = scipy_R.random().as_matrix().astype(np.float32) # (3, 3)
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    if not return_angle:
        return pc_rotated
    else:
        rot_ang = scipy_R.from_matrix(np.transpose(rot_mat)).as_euler('xyz', degrees=True).astype(np.float32) # (3,)
        for aid in range(3):
            if rot_ang[aid] < 0:
                rot_ang[aid] = 360.0 + rot_ang[aid]
        return pc_rotated, rot_ang
    
    
def random_jittering(pc, sigma, bound):
    # pc: (num_points, num_channels)
    # sigma: standard deviation of zero-mean Gaussian noise
    # bound: clip noise values
    # pc_jittered: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert sigma > 0
    assert bound > 0
    gaussian_noises = np.random.normal(0, sigma, size=(num_points, 3)).astype(np.float32) # (num_points, 3)
    bounded_gaussian_noises = np.clip(gaussian_noises, -bound, bound).astype(np.float32) # (num_points, 3)
    if num_channels == 3:
        pc_jittered = pc + bounded_gaussian_noises
    if num_channels > 3:
        xyz = pc[:, 0:3]
        attr = pc[:, 3:]
        pc_jittered = np.concatenate((xyz + bounded_gaussian_noises, attr), axis=1)
    return pc_jittered
    
    
def random_dropout(pc, min_dp_ratio, max_dp_ratio, return_num_dropped=False):
    # pc: (num_points, num_channels)
    # max_dp_ratio: (0, 1)
    # pc_dropped: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_dp_ratio>=0 and min_dp_ratio<=1
    assert max_dp_ratio>=0 and max_dp_ratio<=1
    assert min_dp_ratio <= max_dp_ratio
    dp_ratio = np.random.random() * (max_dp_ratio-min_dp_ratio) + min_dp_ratio
    num_dropped = int(num_points * dp_ratio)
    pc_dropped = pc.copy()
    if num_dropped > 0:
        dp_indices = np.random.choice(num_points, num_dropped, replace=False)
        pc_dropped[dp_indices, :] = pc_dropped[0, :] # all replaced by the first row of "pc"
    if not return_num_dropped:
        return pc_dropped
    else:
        return pc_dropped, num_dropped
    
    
def random_isotropic_scaling(pc, min_s_ratio, max_s_ratio, return_iso_scaling_ratio=False):
    # pc: (num_points, num_channels)
    # pc_iso_scaled: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_s_ratio > 0 and min_s_ratio <= 1
    assert max_s_ratio >= 1
    iso_scaling_ratio = np.random.random() * (max_s_ratio - min_s_ratio) + min_s_ratio
    if num_channels == 3:
        pc_iso_scaled = pc * iso_scaling_ratio
    if num_channels > 3:
        xyz = pc[:, 0:3]
        attr = pc[:, 3:]
        pc_iso_scaled = np.concatenate((xyz * iso_scaling_ratio, attr), axis=1)
    if not return_iso_scaling_ratio:
        return pc_iso_scaled
    else:
        return pc_iso_scaled, iso_scaling_ratio
    
    
def random_anisotropic_scaling(pc, min_s_ratio, max_s_ratio, return_aniso_scaling_ratio=False):
    # pc: (num_points, num_channels)
    # pc_aniso_scaled: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_s_ratio > 0 and min_s_ratio <= 1
    assert max_s_ratio >= 1
    aniso_scaling_ratio = (np.random.random(3) * (max_s_ratio - min_s_ratio) + min_s_ratio).astype('float32')
    pc_aniso_scaled = pc.copy()
    pc_aniso_scaled[:, 0] *= aniso_scaling_ratio[0]
    pc_aniso_scaled[:, 1] *= aniso_scaling_ratio[1]
    pc_aniso_scaled[:, 2] *= aniso_scaling_ratio[2]
    if not return_aniso_scaling_ratio:
        return pc_aniso_scaled
    else:
        return pc_aniso_scaled, aniso_scaling_ratio


def random_translation(pc, max_offset, return_offset=False):
    # pc: (num_points, num_channels)
    # pc_translated: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert max_offset > 0
    offset = np.random.uniform(low=-max_offset, high=max_offset, size=[3]).astype('float32')
    pc_translated = pc.copy()
    pc_translated[:, 0] += offset[0]
    pc_translated[:, 1] += offset[1]
    pc_translated[:, 2] += offset[2]
    if not return_offset:
        return pc_translated
    else:
        return pc_translated, offset
    
    
def centroid_normalization(pc):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - np.mean(xyz, axis=0)
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized
    
    
def bounding_box_normalization(pc):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - (np.min(xyz, axis=0) + np.max(xyz, axis=0))/2
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized


def random_shuffling(pc):
    # pc: (num_points, num_channels)
    # pc_shuffled: (num_points, num_channels)
    num_points, num_channels = pc.shape
    idx_shuffled = np.arange(num_points)
    np.random.shuffle(idx_shuffled)
    pc_shuffled = pc[idx_shuffled]
    return pc_shuffled


def unitize_normals(nms_raw):
    # nms_raw: (num_points, 3)
    assert nms_raw.ndim==2 and nms_raw.shape[1]==3
    scaling = ((nms_raw**2).sum(axis=1, keepdims=True) + 1e-8) ** 0.5
    nms_u = nms_raw / scaling
    return nms_u


def compute_normals_distances(nms_1, nms_2):
    # nms_1: [bs, num_pts, 3]
    # nms_2: [bs, num_pts, 3]
    assert nms_1.size(0) == nms_2.size(0)
    assert nms_1.size(1) == nms_2.size(1)
    assert nms_1.size(2)==3 and nms_2.size(2)==3
    bs, num_pts = nms_1.size(0), nms_1.size(1)
    cos_sim = F.cosine_similarity(nms_1, nms_2, dim=-1) # [bs, num_pts]
    dists = (1.0 - torch.abs(cos_sim)) # [bs, num_pts], within the range of (0, 1)
    return dists


def compute_smooth_cross_entropy(pred, label, eps=0.2):
    #  Cross Entropy Loss with Label Smoothing
    # label = label.contiguous().view(-1)
    # pred: [batch_size, num_classes]
    # label: [batch_size]
    num_classes = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, label.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_prb = F.log_softmax(pred, dim=1)
    sce_loss = -(one_hot * log_prb).sum(dim=1).mean()
    return sce_loss


