# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Module defines the base classes and structures for object tracking in YOLO."""
import numpy as np
import cv2

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH  # 导入您的卡尔曼滤波器
from .Opt_Flow import MaskedOpticalFlow


class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """
        Initialize a new STrack instance.
        """
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """Predicts the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track using new detection data and updates its state and attributes."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Returns the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        """Converts bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        """Returns the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """Returns position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning("WARNING ⚠️ `angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """Returns the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """Returns a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """
    BYTETracker with optical flow compensation for camera motion
    """

    def __init__(self, args, frame_rate=30):
        """
        Initialize a BYTETracker instance for object tracking.
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args

        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

        # 光流补偿相关初始化
        self.flow_calculator = MaskedOpticalFlow(
            max_corners=100,
            quality_level=0.01,
            min_distance=10
        )

        # 补偿状态变量
        self.last_velocity = (0, 0)
        self.reference_velocity = None
        self.in_compensation_mode = False
        self.compensation_scale = 0.5  # 补偿系数

        # 卡尔曼滤波参数动态调节
        self.default_match_thresh = getattr(args, 'match_thresh', 0.8)
        self.max_match_thresh = 0.9
        self.min_match_thresh = 0.3
        self.max_noise_scale = 5.0

        # 始终启用增强的卡尔曼滤波参数
        self.kalman_filter.noise_scale_factor = self.max_noise_scale  # 直接设置为最大值

        # 设置较宽松的匹配阈值
        self.args.match_thresh = self.max_match_thresh  # 直接使用最大匹配阈值

        # 记录原始标志
        self.always_adaptive = False  # 新增标志，表示始终使用自适应模式

        # 存储原始参数，用于恢复
        self.default_noise_scale = getattr(self.kalman_filter, 'noise_scale_factor', 1.0)

        # 是否显示补偿前后对比
        self.show_compensation = True
        # 保存最近处理的帧
        self.current_frame = None
        # 保存输出窗口名称
        self.window_name = "BYTETracker Compensation"
        self.use_gmc = False

    def update(self, results, img=None):
        """Updates the tracker with new detections and returns the current list of tracked objects."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 保存当前帧
        if img is not None:
            self.current_frame = img.copy()

        # 获取检测结果
        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        # 存储补偿前的框位置
        pre_compensation_boxes = []

        # 如果提供了图像，计算光流并应用补偿
        flow_result = {"is_valid": False, "translation_vector": (0, 0)}
        if img is not None:
            # 准备检测框格式用于光流掩码
            xyxy_boxes = []
            for i, box in enumerate(bboxes):
                x, y, w, h = box[:4]
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
                xyxy_boxes.append([x1, y1, x2, y2])

            # 计算光流
            flow_result = self.flow_calculator.compute(img, xyxy_boxes)

        # 按阈值分类检测框
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        # 初始化跟踪器
        detections = self.init_track(dets, scores_keep, cls_keep, img)

        # 获取确认和未确认的轨迹
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:

            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 创建跟踪池
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # # 在应用光流补偿前记录所有跟踪框位置
        # if flow_result['is_valid']:
        #     for track in strack_pool:
        #         if track.is_activated:
        #             pre_compensation_boxes.append(track.result.copy())  # 保存补偿前的结果

        # 预测当前位置
        self.multi_predict(strack_pool)

        # 应用光流补偿
        if flow_result['is_valid']:
            self._apply_flow_compensation(flow_result)

        if flow_result['is_valid']:
         for track in strack_pool:
                if track.is_activated:
                     pre_compensation_boxes.append(track.result.copy())  # 保存补偿前的结果

        # 如果有GMC且传入图像，应用GMC（兼容已有GMC）
        """
        if hasattr(self, "gmc") and img is not None and not self.in_compensation_mode and self.use_gmc:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)
        """

        # 第一轮关联：高分检测框
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二轮关联：低分检测框
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 处理未匹配轨迹
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 处理未确认轨迹
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 创建新轨迹
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # 更新轨迹状态
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 更新轨迹列表
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum

        # 获取补偿后的框
        post_compensation_boxes = [x.result for x in self.tracked_stracks if x.is_activated]

        # 如果有图像且有补偿前的框，绘制补偿对比
        if img is not None and len(pre_compensation_boxes) > 0 and self.show_compensation:
            vis_img = self._visualize_compensation(img, np.array(pre_compensation_boxes),
                                                   np.array(post_compensation_boxes))
            cv2.imshow(self.window_name, vis_img)
            cv2.waitKey(50)  # 显示1毫秒

        # 返回标准格式结果
        return np.asarray(post_compensation_boxes, dtype=np.float32)

    def _visualize_compensation(self, img, pre_boxes, post_boxes):
        """
        在图像上绘制补偿前后的边界框
        """
        # 复制图像以免修改原图
        vis_img = img.copy()

        # 绘制补偿前的框（红色）
        if pre_boxes is not None and len(pre_boxes) > 0:
            for box in pre_boxes:
                if len(box) >= 8:  # 标准格式
                    if len(box) == 9:  # 带角度的框
                        cx, cy, w, h, angle = box[:5]
                        # 绘制旋转框（如果需要）
                        rect = ((cx, cy), (w, h), angle * 180 / np.pi if np.isscalar(angle) else angle)
                        points = cv2.boxPoints(rect).astype(np.int32)
                        cv2.polylines(vis_img, [points], True, (0, 0, 255), 2)  # 红色
                    else:
                        x1, y1, x2, y2 = box[:4]
                        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 红色

                    # 绘制ID
                    track_id = int(box[-4])
                    cv2.putText(vis_img, f"ID:{track_id}", (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 绘制补偿后的框（绿色）
        if post_boxes is not None and len(post_boxes) > 0:
            for box in post_boxes:
                if len(box) >= 8:  # 标准格式
                    if len(box) == 9:  # 带角度的框
                        cx, cy, w, h, angle = box[:5]
                        # 绘制旋转框
                        rect = ((cx, cy), (w, h), angle * 180 / np.pi if np.isscalar(angle) else angle)
                        points = cv2.boxPoints(rect).astype(np.int32)
                        cv2.polylines(vis_img, [points], True, (0, 255, 0), 2)  # 绿色
                    else:
                        x1, y1, x2, y2 = box[:4]
                        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绿色

                    # 绘制ID
                    track_id = int(box[-4])
                    cv2.putText(vis_img, f"ID:{track_id}", (int(box[0]), int(box[1]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示信息
        cv2.putText(vis_img, "Pre-compensation (Red)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_img, "Post-compensation (Green)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return vis_img

    def _apply_flow_compensation(self, flow_result):
        """应用光流补偿到跟踪器状态"""
        # 获取当前光流速度
        tx, ty = flow_result["translation_vector"]
        acceleration_threshold = 7.0
        ks = 1

        # 保持增强的卡尔曼滤波参数
        self.always_adaptive=False
        if self.always_adaptive:
            self.kalman_filter.noise_scale_factor = self.max_noise_scale
            self.args.match_thresh = self.max_match_thresh

        # 计算加速度
        if hasattr(self, 'last_velocity') and self.last_velocity != (0, 0):  # 非第一帧
            ax_abs = tx - self.last_velocity[0]
            ay_abs = ty - self.last_velocity[1]
            acceleration_magnitude = np.sqrt(ax_abs ** 2 + ay_abs ** 2)

            # 初始帧处理
            if self.frame_id <= 3:
                self.last_velocity = (tx, ty)
                return

            # 检测突变并启动补偿模式
            if not self.in_compensation_mode and acceleration_magnitude > acceleration_threshold:
                self.reference_velocity = self.last_velocity
                self.in_compensation_mode = True
                LOGGER.info(f"检测到突变! 加速度={acceleration_magnitude:.2f}, 参考速度={self.reference_velocity}")


                #     # 增大卡尔曼滤波过程噪声
                if hasattr(self.kalman_filter, 'noise_scale_factor'):
                       self.kalman_filter.noise_scale_factor = min(
                             self.kalman_filter.noise_scale_factor * 2.0,
                            self.max_noise_scale
                        )
                    # 放宽匹配阈值
                self.args.match_thresh = min(
                        self.args.match_thresh * 1.5,
                        self.max_match_thresh
                     )
                #     LOGGER.info(
                #         f"场景变化: 噪声尺度={self.kalman_filter.noise_scale_factor:.2f}, 匹配阈值={self.args.match_thresh:.2f}")

                # 检测突变并启动补偿模式

            # 执行补偿
            if self.in_compensation_mode:
                dx_compensation = 0
                dy_compensation = 0

                # X方向补偿计算
                if abs(tx - self.reference_velocity[0]) > 5.0:
                    if tx * self.reference_velocity[0] >= 0:  # 方向相同
                        dx_compensation = (tx - self.reference_velocity[0]) * ks
                    else:  # 方向改变
                        if tx > 0:
                            dx_compensation = (tx + abs(self.reference_velocity[0])) * ks
                        else:
                            dx_compensation = (tx - self.reference_velocity[0]) * ks

                # Y方向补偿
                if abs(ty - self.reference_velocity[1]) > 5.0:
                    if ty * self.reference_velocity[1] >= 0:
                        dy_compensation = (ty - self.reference_velocity[1]) * ks
                    else:
                        if ty > 0:
                            dy_compensation = (ty + abs(self.reference_velocity[1])) * ks
                        else:
                            dy_compensation = (ty - self.reference_velocity[1]) * ks

                # 应用补偿到所有轨迹
                if abs(dx_compensation) > 0 or abs(dy_compensation) > 0:
                    LOGGER.info(f"应用补偿: dx={dx_compensation:.2f}, dy={dy_compensation:.2f}")
                    for track in self.tracked_stracks + self.lost_stracks:
                        if track.mean is not None:

                            track.mean[0] += dx_compensation  # 位置x
                            track.mean[1] += dy_compensation  # 位置y



                # 检查是否退出补偿模式
                velocity_close = (abs(tx - self.reference_velocity[0]) < 2.0 and
                                  abs(ty - self.reference_velocity[1]) < 2.0)

                current_ax_abs = abs(tx - self.reference_velocity[0]) - abs(
                    self.last_velocity[0] - self.reference_velocity[0])
                current_ay_abs = abs(ty - self.reference_velocity[1]) - abs(
                    self.last_velocity[1] - self.reference_velocity[1])
                current_acc_magnitude = np.sqrt(current_ax_abs ** 2 + current_ay_abs ** 2)



                if velocity_close or current_acc_magnitude < 0.3 * acceleration_threshold:
                    LOGGER.info("退出补偿模式，速度已恢复稳定")
                    self.in_compensation_mode = False

                    """

                    # 速度补偿
                    x = self.last_velocity[0] - self.reference_velocity[0]
                    y = self.last_velocity[1] - self.reference_velocity[1]

                    if abs(x) > 10:
                        LOGGER.info(f"X速度补偿: {x * ks:.2f}")
                        for track in self.tracked_stracks + self.lost_stracks:
                            if track.mean is not None:
                                track.mean[4] += x * ks  # 速度x

                    if abs(y) > 10:
                        LOGGER.info(f"Y速度补偿: {y * ks:.2f}")
                        for track in self.tracked_stracks + self.lost_stracks:
                            if track.mean is not None:
                                track.mean[5] += y * ks  # 速度y
                 """

                    # 场景稳定时恢复默认参数

                    self.args.match_thresh = max(
                             self.args.match_thresh / 1.5,
                            self.default_match_thresh
                         )
                    if hasattr(self.kalman_filter, 'noise_scale_factor'):
                             self.kalman_filter.noise_scale_factor = self.default_noise_scale

                    self.reference_velocity = None

        # 更新速度记录
        self.last_velocity = (tx, ty)

    # 其他方法保持不变...

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """Initializes object tracking with given detections, scores, and class labels using the STrack algorithm."""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IoU and optionally fuses scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Resets the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Resets the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

        # 重置光流状态
        self.flow_calculator.reset()
        self.last_velocity = (0, 0)
        self.reference_velocity = None
        self.in_compensation_mode = False

        # 保持自适应卡尔曼滤波
        if self.always_adaptive:
            self.kalman_filter.noise_scale_factor = self.max_noise_scale
            self.args.match_thresh = self.max_match_thresh

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combines two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Filters out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Removes duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb


# 可视化函数
def visualize_compensation(img, pre_boxes, post_boxes):
    """
    在图像上绘制补偿前后的边界框

    Args:
        img: 输入图像
        pre_boxes: 补偿前的边界框 [x1,y1,x2,y2,track_id,score,class,idx] 或 [cx,cy,w,h,angle,track_id,score,class,idx]
        post_boxes: 补偿后的边界框

    Returns:
        带有可视化结果的图像
    """
    # 复制图像以免修改原图
    vis_img = img.copy()

    # 绘制补偿前的框（红色）
    if pre_boxes is not None:
        for box in pre_boxes:
            if len(box) >= 8:  # 标准格式
                if len(box) == 9:  # 带角度的框
                    cx, cy, w, h, angle = box[:5]
                    # 绘制旋转框（如果需要）
                    rect = ((cx, cy), (w, h), angle * 180 / np.pi if np.isscalar(angle) else angle)
                    points = cv2.boxPoints(rect).astype(np.int32)
                    cv2.polylines(vis_img, [points], True, (0, 0, 255), 2)  # 红色
                else:
                    x1, y1, x2, y2 = box[:4]
                    cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 红色

                # 绘制ID
                track_id = int(box[-4])
                cv2.putText(vis_img, f"ID:{track_id}", (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 绘制补偿后的框（绿色）
    if post_boxes is not None:
        for box in post_boxes:
            if len(box) >= 8:  # 标准格式
                if len(box) == 9:  # 带角度的框
                    cx, cy, w, h, angle = box[:5]
                    # 绘制旋转框
                    rect = ((cx, cy), (w, h), angle * 180 / np.pi if np.isscalar(angle) else angle)
                    points = cv2.boxPoints(rect).astype(np.int32)
                    cv2.polylines(vis_img, [points], True, (0, 255, 0), 2)  # 绿色
                else:
                    x1, y1, x2, y2 = box[:4]
                    cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绿色

                # 绘制ID
                track_id = int(box[-4])
                cv2.putText(vis_img, f"ID:{track_id}", (int(box[0]), int(box[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return vis_img