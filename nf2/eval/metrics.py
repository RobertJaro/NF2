from nf2.evaluation.output_metrics import metric_mapping

METRICS = metric_mapping


def get_metric(name):
    return METRICS[name]


def compute_metrics(metric_names, state):
    return {name: get_metric(name)(**state) for name in metric_names}
