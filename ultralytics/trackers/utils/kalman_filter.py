
# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """
    A KalmanFilterXYAH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a simple Kalman filter for tracking bounding boxes in image space. The 8-dimensional state space
    (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect ratio a, height h, and their
    respective velocities. Object motion follows a constant velocity model, and bounding box location (x, y, a, h) is
    taken as a direct observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Creates a track from an unassociated measurement.
        predict: Runs the Kalman filter prediction step.
        project: Projects the state distribution to measurement space.
        multi_predict: Runs the Kalman filter prediction step (vectorized version).
        update: Runs the Kalman filter correction step.
        gating_distance: Computes the gating distance between state distribution and measurements.

    Notes:
        已集成 NSA（创新一致性驱动的自适应噪声缩放）机制：
        - 通过创新窗口的经验协方差与理论创新协方差迹比率估计目标尺度
        - 对过程噪声进行平滑缩放（观测噪声保持不变），提高模型对非稳态运动/噪声变化的鲁棒性
    """

    def __init__(self):
        """
        Initialize Kalman filter model matrices with motion and observation uncertainty weights.

        The Kalman filter is initialized with an 8-dimensional state space (x, y, a, h, vx, vy, va, vh), where (x, y)
        represents the bounding box center position, 'a' is the aspect ratio, 'h' is the height, and their respective
        velocities are (vx, vy, va, vh). The filter uses a constant velocity model for object motion and a linear
        observation model for bounding box location.
        """
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty (relative to object size)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        # NSA 自适应相关参数（如需调整，可在实例化后通过属性修改）
        self.window_size = 10
        self.adaptive_factor = 0.3
        self.min_scale_factor = 1.0
        self.max_scale_factor = 10.0

        # 存储创新序列和当前噪声尺度因子
        self.innovation_history = []
        self.noise_scale_factor = 1.0
        self.last_innovation_cov = None

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create a track from an unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8-dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))

        # 重置自适应相关状态
        self.innovation_history = []
        self.noise_scale_factor = 1.0
        self.last_innovation_cov = None

        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state.
        """
        # 过程噪声按 NSA 缩放
        sf = self.noise_scale_factor
        std_pos = [
            self._std_weight_position * mean[3] * sf,
            self._std_weight_position * mean[3] * sf,
            1e-2 * sf,
            self._std_weight_position * mean[3] * sf,
        ]
        std_vel = [
            self._std_weight_velocity * mean[3] * sf,
            self._std_weight_velocity * mean[3] * sf,
            1e-5 * sf,
            self._std_weight_velocity * mean[3] * sf,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        # 观测噪声不随 NSA 缩放，保持一致性检测灵敏度
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        self.last_innovation_cov = innovation_cov.copy()

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step for multiple object states (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean matrix and covariance matrix of the predicted states.
        """
        sf = self.noise_scale_factor
        std_pos = [
            self._std_weight_position * mean[:, 3] * sf,
            self._std_weight_position * mean[:, 3] * sf,
            1e-2 * np.ones_like(mean[:, 3]) * sf,
            self._std_weight_position * mean[:, 3] * sf,
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3] * sf,
            self._std_weight_velocity * mean[:, 3] * sf,
            1e-5 * np.ones_like(mean[:, 3]) * sf,
            self._std_weight_velocity * mean[:, 3] * sf,
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, a, h).

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        # 保存创新并做 NSA 自适应
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)
        if (self.last_innovation_cov is not None) and (len(self.innovation_history) >= self.window_size):
            self.adjust_noise_scale()

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def adjust_noise_scale(self):
        """
        基于创新序列经验协方差与理论创新协方差的迹比率，平滑调节过程噪声缩放因子。
        """
        try:
            innovations = np.asarray(self.innovation_history)
            if innovations.ndim != 2:
                return
            empirical_cov = np.cov(innovations.T)
            if self.last_innovation_cov is None:
                return
            if empirical_cov.shape != self.last_innovation_cov.shape:
                return

            denom = max(float(np.trace(self.last_innovation_cov)), 1e-12)
            trace_ratio = float(np.trace(empirical_cov) / denom)
            target_scale = float(np.sqrt(max(trace_ratio, 1.0)))

            self.noise_scale_factor = (
                self.noise_scale_factor * (1.0 - self.adaptive_factor) +
                target_scale * self.adaptive_factor
            )
            self.noise_scale_factor = float(
                np.clip(self.noise_scale_factor, self.min_scale_factor, self.max_scale_factor)
            )
        except Exception:
            # 出现异常时保持当前尺度不变
            pass

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If `only_position` is False, the chi-square
        distribution has 4 degrees of freedom, otherwise 2.

        Args:
            mean (ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (ndarray): An (N, 4) matrix of N measurements (x, y, a, h).
            only_position (bool): If True, distance computation uses only (x, y).
            metric (str): 'gaussian' for squared Euclidean or 'maha' for squared Mahalanobis.

        Returns:
            (np.ndarray): Returns an array of length N with squared distances.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # squared Mahalanobis
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    A KalmanFilterXYWH class for tracking bounding boxes in image space using a Kalman filter.

    State: (x, y, w, h, vx, vy, vw, vh)
    """

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, w, h).

        Returns:
            (tuple[ndarray, ndarray]): mean (8,), covariance (8, 8)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))

        # 重置 NSA 状态
        self.innovation_history = []
        self.noise_scale_factor = 1.0
        self.last_innovation_cov = None

        return mean, covariance

    def predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step for XYWH.
        """
        sf = self.noise_scale_factor
        std_pos = [
            self._std_weight_position * mean[2] * sf,
            self._std_weight_position * mean[3] * sf,
            self._std_weight_position * mean[2] * sf,
            self._std_weight_position * mean[3] * sf,
        ]
        std_vel = [
            self._std_weight_velocity * mean[2] * sf,
            self._std_weight_velocity * mean[3] * sf,
            self._std_weight_velocity * mean[2] * sf,
            self._std_weight_velocity * mean[3] * sf,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance) -> tuple:
        """
        Project state distribution to measurement space for XYWH.
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        self.last_innovation_cov = innovation_cov.copy()

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version) for XYWH.
        """
        sf = self.noise_scale_factor
        std_pos = [
            self._std_weight_position * mean[:, 2] * sf,
            self._std_weight_position * mean[:, 3] * sf,
            self._std_weight_position * mean[:, 2] * sf,
            self._std_weight_position * mean[:, 3] * sf,
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2] * sf,
            self._std_weight_velocity * mean[:, 3] * sf,
            self._std_weight_velocity * mean[:, 2] * sf,
            self._std_weight_velocity * mean[:, 3] * sf,
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement) -> tuple:
        """
        Run Kalman filter correction step (inherits NSA behavior from parent).
        """
        return super().update(mean, covariance, measurement)
