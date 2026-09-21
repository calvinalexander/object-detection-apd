# ============================================================
# CLI ENTRY
# ============================================================
import argparse
import multiprocessing
from apd_config import config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--interval",
        type=int,
        default=5,
        help="detik interval polling"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if config.APD.MODE == 'stream':
        from apd_stream_service import StreamSupervisor
        supervisor = StreamSupervisor(poll_interval=args.interval)
    elif config.APD.MODE == 'recording':
        from apd_recording_service import RecordingSupervisor
        supervisor = RecordingSupervisor(poll_interval=args.interval)

    if supervisor is None:
        raise RuntimeError("No supervisor selected (check config.stream_mode)")

    supervisor.run()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
