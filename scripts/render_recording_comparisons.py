import os, subprocess

videos = [
    # Master
    ("/data/adin/sppg-ai/cp/Recording 2026-09-09 104135.mp4", "artifacts/compare.mp4"),
    # Batch 1
    ("data/recording/Recording 2026-09-11 113142.mp4", "artifacts/compare_rec113142.mp4"),
    ("data/recording/Recording 2026-09-11 113959.mp4", "artifacts/compare_rec113959.mp4"),
    ("data/recording/Screen Recording 2026-09-11 113644.mp4", "artifacts/compare_rec113644.mp4"),
    # Batch 2
    ("data/recording/batch_2/Recording 2026-09-11 133700.mp4", "artifacts/compare_b2_rec133700.mp4"),
    ("data/recording/batch_2/Recording 2026-09-11 133958.mp4", "artifacts/compare_b2_rec133958.mp4"),
    ("data/recording/batch_2/Recording 2026-09-11 134226.mp4", "artifacts/compare_b2_rec134226.mp4"),
    # Batch 3
    ("data/recording/batch_3/Recording 2026-09-11 135526.mp4", "artifacts/compare_b3_rec135526.mp4"),
    ("data/recording/batch_3/Recording 2026-09-11 135711.mp4", "artifacts/compare_b3_rec135711.mp4"),
    ("data/recording/batch_3/Recording 2026-09-11 140115.mp4", "artifacts/compare_b3_rec140115.mp4"),
]

for vid_path, out_path in videos:
    cmd = [
        "python3", "scripts/55_compare_video.py",
        "--video", vid_path,
        "--edge-pose", "yolo11s-pose",
        "--cls-model", "effnet_b1",
        "--band-absent", "0.10",
        "--band-present", "0.90",
        "--clean", "0",
        "--seconds", "10.0",
        "--out", out_path
    ]
    print(f"\n================ UPDATING RECORDING COMPARISON: {out_path} ================")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr and "Warning" not in res.stderr:
        print(res.stderr)
