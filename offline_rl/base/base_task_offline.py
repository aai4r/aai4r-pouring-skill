

from tasks.base.base_task import BaseTask


class BaseTaskOffline(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def save_record(self):
        pass

    def load_record(self):
        pass



