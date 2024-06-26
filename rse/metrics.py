from torchmetrics import JaccardIndex as IoU, Accuracy

class CmIoU(IoU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, average='macro', **kwargs)

class NmIoU(IoU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, average='micro', **kwargs)

Acc = Accuracy

