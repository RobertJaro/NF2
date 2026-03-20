from nf2.output.checkpoint import NF2Checkpoint


def open_checkpoint(path, device=None):
    return NF2Checkpoint(path, device=device)
