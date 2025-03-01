import datetime
from tensorboardX import SummaryWriter


class MetricsTracker:

    def __init__(self, logdir: str, prefix: str):
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # now = "debug"
        self.writer = SummaryWriter(logdir + "/" + prefix + now)

    def register_scalars(self, step: int, metrics_dict: dict[str, float]):
        for name, value in metrics_dict.items():
            self.writer.add_scalar(name, value, step)

    def register_figure(self, step: int, fig):
        self.writer.add_figure("figure", fig, step)
