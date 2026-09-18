"""设备管理器类"""

import torch
import gc
import logging

# 全局logger
logger = logging.getLogger(__name__)

# ==================== 设备管理器类 ====================
class DeviceManager:
    """设备管理器，统一管理所有数据的设备移动"""

    def __init__(self, target_device=None):
        self.target_device = target_device if target_device != "auto"  else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"设备管理器初始化，目标设备: {self.target_device}")

    def move_tensor(self, tensor):
        """移动张量到目标设备"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.target_device)
        return tensor

    def move_data(self, data):
        """移动Data对象到目标设备"""
        if hasattr(data, 'to'):
            return data.to(self.target_device)
        return data

    def move_hetero_data(self, hetero_data):
        """移动HeteroData对象到目标设备"""
        try:
            # 先尝试使用内置方法
            if hasattr(hetero_data, 'to'):
                hetero_data = hetero_data.to(self.target_device)
                logger.info(f"使用内置方法移动HeteroData到 {self.target_device}")
                return hetero_data

            # 手动移动节点特征
            for node_type in hetero_data.node_types:
                if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
                    hetero_data[node_type].x = self.move_tensor(hetero_data[node_type].x)

            # 手动移动边索引
            for edge_type in hetero_data.edge_types:
                if hasattr(hetero_data[edge_type], 'edge_index') and hetero_data[edge_type].edge_index is not None:
                    hetero_data[edge_type].edge_index = self.move_tensor(hetero_data[edge_type].edge_index)

            logger.info(f"手动移动HeteroData到 {self.target_device}")
            return hetero_data

        except Exception as e:
            logger.error(f"移动HeteroData失败: {e}")
            return hetero_data

    def move_model(self, model):
        """移动模型到目标设备"""
        return model.to(self.target_device)

    def ensure_all_on_device(self, model, sl_data, kg_data):
        """确保所有组件都在目标设备上"""
        # 检查并移动模型
        model_device = next(model.parameters()).device
        if model_device.type != self.target_device.split(':')[0]:
            logger.warning(f"模型在 {model_device}，移动到 {self.target_device}")
            model = self.move_model(model)

        # 检查并移动SL数据
        if hasattr(sl_data, 'x'):
            if sl_data.x.device.type != self.target_device.split(':')[0]:
                logger.warning(f"SL数据在 {sl_data.x.device}，移动到 {self.target_device}")
                sl_data = self.move_data(sl_data)

        # 检查并移动知识图谱数据
        for node_type in kg_data.node_types:
            if hasattr(kg_data[node_type], 'x') and kg_data[node_type].x is not None:
                if kg_data[node_type].x.device.type != self.target_device.split(':')[0]:
                    logger.warning(f"{node_type}特征在 {kg_data[node_type].x.device}，移动到 {self.target_device}")
                    kg_data = self.move_hetero_data(kg_data)
                    break

        return model, sl_data, kg_data

    def clear_cache(self):
        """清理GPU缓存"""
        if 'cuda' in self.target_device:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()