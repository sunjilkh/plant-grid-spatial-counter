import csv

with open("event_audit.csv") as f:
    rows = list(csv.DictReader(f))

print("=== EVENT AUDIT (%d events) ===" % len(rows))
print("  %3s  %5s  %-6s %-8s %5s  %5s  %5s  %6s" % (
    "#", "Frame", "Lane", "Reason", "TrkID", "AncX", "AncY", "Conf"))
print("-" * 65)
for r in rows:
    print("  %3s  %5s  %-6s %-8s %5s  %5s  %5s  %6.3f" % (
        r["count_at_event"], r["frame"], r["lane"], r["event_reason"],
        r["track_id"], r["anchor_x"], r["anchor_y"], float(r["conf"])))

print()

with open("missed_tracks.csv") as f:
    missed = list(csv.DictReader(f))

print("=== MISSED TRACKS (%d total) ===" % len(missed))
print("  %5s  %6s  %5s  %5s  %5s  %5s  %-12s  %-6s  %7s" % (
    "TrkID", "Frames", "First", "Last", "MinY", "MaxY", "CrossedBand", "Lane", "AvgConf"))
print("-" * 80)
for m in missed:
    print("  %5s  %6s  %5s  %5s  %5s  %5s  %-12s  %-6s  %7.4f" % (
        m["track_id"], m["frames_seen"], m["first_frame"], m["last_frame"],
        m["min_y"], m["max_y"], m["crossed_band"], m["lane"], float(m["avg_conf"])))

# --- duplicate check ---
print()
print("=== SUSPECTED DUPLICATES (same lane, dy<20, df<5) ===")
found = 0
for i, a in enumerate(rows):
    for b in rows[i+1:]:
        if a["lane"] != b["lane"]:
            continue
        df = abs(int(a["frame"]) - int(b["frame"]))
        dy = abs(int(a["anchor_y"]) - int(b["anchor_y"]))
        if df <= 5 and dy <= 20:
            print("  ! Event #%s and #%s: lane=%s  dy=%d  df=%d" % (
                a["count_at_event"], b["count_at_event"], a["lane"], dy, df))
            found += 1
if found == 0:
    print("  None detected — dedup is clean.")
