import cv2
import numpy as np
import math


class MaskedOpticalFlow:
    """
    掩码光流计算类 - 用于无人机场景下目标跟踪，分离目标运动与背景运动
    """

    def __init__(self, max_corners=100, quality_level=0.01, min_distance=10,
                 block_size=3, detection_dilation=10, ransac_threshold=3.0):
        """
        初始化光流计算器

        参数:
            max_corners: 特征点最大数量
            quality_level: 特征点质量阈值
            min_distance: 特征点最小距离
            block_size: 特征点计算窗口大小
            detection_dilation: 检测框膨胀像素数
            ransac_threshold: RANSAC算法阈值（用于稳定位移估计）
        """
        # 特征点检测参数
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.detection_dilation = detection_dilation
        self.ransac_threshold = ransac_threshold

        # 状态变量
        self.prev_gray = None
        self.prev_points = None
        self.mask_border_ratio = 0.05

        # 光流计算参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def reset(self):
        """重置计算状态"""
        self.prev_gray = None
        self.prev_points = None

    def _create_detection_mask(self, shape, detections):
        """创建检测框掩码 - 将检测到的目标区域排除"""
        h, w = shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        if detections is None or len(detections) == 0:
            return mask

        for det in detections:
            if len(det) == 4:
                x1, y1, x2, y2 = map(int, det)

                # 膨胀检测框
                x1 = max(0, x1 - self.detection_dilation)
                y1 = max(0, y1 - self.detection_dilation)
                x2 = min(w, x2 + self.detection_dilation)
                y2 = min(h, y2 + self.detection_dilation)

                # 将检测框区域置为0
                mask[y1:y2, x1:x2] = 0

        return mask

    def _create_border_mask(self, shape):
        """创建边缘掩码 - 排除图像边缘区域"""
        h, w = shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        border_h = int(h * self.mask_border_ratio)
        border_w = int(w * self.mask_border_ratio)

        # 设置边缘区域为0
        mask[:border_h, :] = 0
        mask[h - border_h:, :] = 0
        mask[:, :border_w] = 0
        mask[:, w - border_w:] = 0

        return mask

    def _create_combined_mask(self, shape, detections):
        """创建组合掩码（边缘+检测框）"""
        detection_mask = self._create_detection_mask(shape, detections)
        border_mask = self._create_border_mask(shape)
        return cv2.bitwise_and(detection_mask, border_mask)

    def compute(self, frame, detections=None):
        """
        计算光流，排除检测框区域，只计算背景运动的平移向量和速度

        参数:
            frame: 输入图像
            detections: 检测框列表 [[x1, y1, x2, y2], ...]

        返回:
            result: 包含以下字段的字典:
                - masked_frame: 应用掩码后的图像
                - detection_mask: 检测框掩码
                - combined_mask: 组合掩码
                - flow_vectors: 光流向量列表 [(x1, y1, dx, dy), ...]
                - feature_points: 当前帧特征点
                - translation_vector: (dx, dy) 平移向量
                - speed: 速度大小
                - is_valid: 结果是否有效
        """
        # 默认结果字典
        result = {
            'masked_frame': None,
            'detection_mask': None,
            'combined_mask': None,
            'flow_vectors': [],
            'feature_points': None,
            'translation_vector': (0, 0),
            'speed': 0,
            'is_valid': False
        }

        if frame is None:
            return result

        # 转为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            color_frame = frame.copy()
        else:
            gray = frame.copy()
            color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 创建掩码
        detection_mask = self._create_detection_mask(frame.shape, detections)
        combined_mask = self._create_combined_mask(frame.shape, detections)

        # 创建掩码图像（将检测框区域置为0）
        masked_frame = color_frame.copy()
        masked_frame[detection_mask == 0] = 0

        # 更新结果
        result['masked_frame'] = masked_frame
        result['detection_mask'] = detection_mask
        result['combined_mask'] = combined_mask

        # 首次调用或图像尺寸变化，初始化状态
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.reset()
            self.prev_gray = gray.copy()

            # 在有效区域检测特征点
            try:
                self.prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    blockSize=self.block_size,
                    mask=combined_mask
                )

                if self.prev_points is not None:
                    result['feature_points'] = self.prev_points.reshape(-1, 2)
            except Exception as e:
                print(f"特征点检测错误: {e}")

            return result

        # 确保有特征点可跟踪
        if self.prev_points is None or len(self.prev_points) < 10:
            # 重新检测特征点
            try:
                self.prev_points = cv2.goodFeaturesToTrack(
                    self.prev_gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    blockSize=self.block_size,
                    mask=combined_mask
                )

                if self.prev_points is None or len(self.prev_points) < 10:
                    self.prev_gray = gray.copy()
                    return result
            except Exception as e:
                print(f"特征点重新检测错误: {e}")
                self.prev_gray = gray.copy()
                return result

        # 计算光流
        try:
            curr_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray,
                self.prev_points, None,
                **self.lk_params
            )
        except cv2.error as e:
            print(f"光流计算错误: {e}")
            self.reset()
            self.prev_gray = gray.copy()
            return result

        # 筛选有效点
        status = status.ravel()
        valid_indices = np.where(status == 1)[0]

        if len(valid_indices) < 10:
            self.prev_gray = gray.copy()
            # 重新检测特征点
            self.prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                mask=combined_mask
            )

            if self.prev_points is not None:
                result['feature_points'] = self.prev_points.reshape(-1, 2)

            return result

        prev_valid_points = self.prev_points[valid_indices].reshape(-1, 2)
        curr_valid_points = curr_points[valid_indices].reshape(-1, 2)

        # 额外过滤：检查当前点是否在目标区域外
        valid_mask = np.zeros(len(curr_valid_points), dtype=bool)
        for i, (x, y) in enumerate(curr_valid_points):
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < gray.shape[1] and 0 <= y_int < gray.shape[0]:
                if detection_mask[y_int, x_int] == 255:  # 点在有效区域
                    valid_mask[i] = True

        # 应用额外过滤
        if np.sum(valid_mask) < 10:
            self.prev_gray = gray.copy()
            # 重新检测特征点
            self.prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                mask=combined_mask
            )

            if self.prev_points is not None:
                result['feature_points'] = self.prev_points.reshape(-1, 2)

            return result

        prev_valid_points = prev_valid_points[valid_mask]
        curr_valid_points = curr_valid_points[valid_mask]

        # 计算光流向量
        flow_vectors = []
        for i in range(len(prev_valid_points)):
            x1, y1 = prev_valid_points[i]
            x2, y2 = curr_valid_points[i]
            dx, dy = x2 - x1, y2 - y1
            flow_vectors.append((x1, y1, dx, dy))

        result['flow_vectors'] = flow_vectors
        result['feature_points'] = curr_valid_points

        # 使用RANSAC估计仿射变换 - 这比简单平均所有点位移更稳健
        # 仿射变换矩阵能提供一个整体性的位移估计，减少异常点的影响
        try:
            affine_matrix, inliers = cv2.estimateAffinePartial2D(
                prev_valid_points, curr_valid_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold
            )
        except Exception as e:
            print(f"仿射变换估计错误: {e}")
            affine_matrix = None

        if affine_matrix is not None:
            # 从仿射变换矩阵中提取平移分量
            # affine_matrix[:, 2]包含x和y方向的平移量
            translation = affine_matrix[:, 2]

            # 计算速度和方向
            tx, ty = float(translation[0]), float(translation[1])
            speed = math.sqrt(tx ** 2 + ty ** 2)

            # 更新结果
            result['translation_vector'] = (tx, ty)
            result['speed'] = float(speed)
            result['is_valid'] = True
        else:
            # 如果仿射变换失败，可以回退到计算平均位移
            if len(flow_vectors) > 0:
                dx_sum = sum(dx for _, _, dx, _ in flow_vectors)
                dy_sum = sum(dy for _, _, _, dy in flow_vectors)
                avg_dx = dx_sum / len(flow_vectors)
                avg_dy = dy_sum / len(flow_vectors)
                speed = math.sqrt(avg_dx ** 2 + avg_dy ** 2)

                result['translation_vector'] = (avg_dx, avg_dy)
                result['speed'] = float(speed)
                result['is_valid'] = True

        # 更新状态
        self.prev_gray = gray.copy()

        # 使用当前有效点作为下一次的跟踪点，或重新检测
        if len(curr_valid_points) < self.max_corners // 2:
            try:
                self.prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    blockSize=self.block_size,
                    mask=combined_mask
                )
            except Exception:
                self.prev_points = curr_valid_points.reshape(-1, 1, 2)
        else:
            self.prev_points = curr_valid_points.reshape(-1, 1, 2)

        return result