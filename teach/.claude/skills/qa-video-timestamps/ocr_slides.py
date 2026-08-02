"""Sample frames from a slide-recording video and OCR the footer slide counter.

Outputs TSV: time_seconds<TAB>slide_number<TAB>total (or '-' when no counter read).
Usage: python ocr_slides.py <video> <interval_seconds>
"""
import re
import sys

import cv2
import objc  # noqa: F401  (ensures pyobjc loaded before Vision)
import Quartz
import Vision
from Foundation import NSData


def ocr_png_bytes(png_bytes):
    data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    if src is None:
        return ""
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return ""
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    ok, _err = handler.performRequests_error_([request], None)
    if not ok or not request.results():
        return ""
    return " ".join(
        r.topCandidates_(1)[0].string() for r in request.results() if r.topCandidates_(1)
    )


def main():
    video_path, interval = sys.argv[1], float(sys.argv[2])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    counter_re = re.compile(r"(\d+)\s*/\s*(\d+)")

    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        # The recording places slides in the left ~73% of the frame with the
        # webcam on the right; the "Lambert n/N" counter sits bottom-right of
        # the slide box. Also try a full-frame corner crop in case a segment
        # shows the deck fullscreen.
        m = None
        for y0, y1, x0, x1 in ((0.76, 0.90, 0.50, 0.76), (0.88, 1.0, 0.70, 1.0)):
            crop = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            ok, png = cv2.imencode(".png", crop)
            text = ocr_png_bytes(png.tobytes()) if ok else ""
            m = counter_re.search(text)
            if m:
                break
        if m:
            print(f"{t:.0f}\t{m.group(1)}\t{m.group(2)}", flush=True)
        else:
            print(f"{t:.0f}\t-\t-", flush=True)
        t += interval
    cap.release()


if __name__ == "__main__":
    main()
