from ..registry import DETECTORS
from .single_stage import SingleStageDetector

#装饰器装饰RetinaNet的功能，继承了单阶段的模型
@DETECTORS.register_module
class RetinaNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
