"""Refine question-slide timestamps to 1s resolution.

For each target slide: take the first 5s-sample where the footer counter shows
that slide, then OCR every second in the preceding window to find the exact
switch second. Prints: slide<TAB>seconds<TAB>MM:SS.

Usage: python refine_timestamps.py <video> <tsv> <expected_total> <slide,...>
"""
import sys

import cv2

from ocr_slides import ocr_png_bytes
import re

counter_re = re.compile(r"(\d+)\s*/\s*(\d+)")


def read_counter(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = frame.shape[:2]
    for y0, y1, x0, x1 in ((0.76, 0.90, 0.50, 0.76), (0.88, 1.0, 0.70, 1.0)):
        crop = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        ok, png = cv2.imencode(".png", crop)
        m = counter_re.search(ocr_png_bytes(png.tobytes()) if ok else "")
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def main():
    video, tsv, expected_total = sys.argv[1], sys.argv[2], int(sys.argv[3])
    targets = [int(x) for x in sys.argv[4].split(",")]

    readings = []
    for line in open(tsv):
        t, n, total = line.strip().split("\t")
        if n == "-" or int(total) != expected_total:
            continue
        readings.append((float(t), int(n)))

    cap = cv2.VideoCapture(video)
    for target in targets:
        t0 = None
        for i, (t, n) in enumerate(readings):
            if n != target:
                continue
            nxt = readings[i + 1][1] if i + 1 < len(readings) else n
            if abs(nxt - n) <= 1:
                t0 = t
                break
        if t0 is None:
            print(f"{target}\tMISSING\t-")
            continue
        # The switch happened in (t0 - 5, t0]; find the first second showing it.
        switch = int(t0)
        for t in range(int(t0) - 4, int(t0) + 1):
            r = read_counter(cap, t)
            if r and r[0] == target and r[1] == expected_total:
                switch = t
                break
        print(f"{target}\t{switch}\t{switch // 60}:{switch % 60:02d}")
    cap.release()


if __name__ == "__main__":
    main()
