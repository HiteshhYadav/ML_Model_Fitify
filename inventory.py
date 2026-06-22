import os, collections

path = r"C:\Users\yadav\.cache\kagglehub\datasets\philosopher0808\gym-workoutexercises-video\versions\1"

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
counts = collections.Counter()
sizes  = collections.defaultdict(list)

for root, _, files in os.walk(path):
    cls = os.path.basename(root)
    for f in files:
        if os.path.splitext(f)[1].lower() in VIDEO_EXT:
            counts[cls] += 1
            sizes[cls].append(os.path.getsize(os.path.join(root, f)))

print(f"{'exercise':30} {'n_clips':>7} {'avg_MB':>8}")
for cls, n in counts.most_common():
    avg_mb = sum(sizes[cls]) / n / 1e6
    print(f"{cls:30} {n:>7} {avg_mb:>8.1f}")
print("\nTOTAL clips:", sum(counts.values()), "| folders:", len(counts))
