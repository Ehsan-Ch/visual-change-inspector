import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps
from .core import Config, inspect
from .report import example, write_report


def load_pair(before_path, after_path, max_side=720):
    images = []
    for path in (before_path, after_path):
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert('RGB'))
    if images[0].size != images[1].size:
        raise ValueError('Images must have equal dimensions after EXIF orientation correction.')
    w, h = images[0].size
    scale = min(1, max_side/max(w, h))
    size = (round(w*scale), round(h*scale))
    return [np.array(im.resize(size, Image.Resampling.LANCZOS)) for im in images]


def main():
    parser = argparse.ArgumentParser(description='Compare two images locally, with alignment quality checks.')
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('benchmark', help='Run the controlled benchmark (requires benchmark extra)')
    b.add_argument('--out', default='reports/benchmark')
    c = sub.add_parser('compare', help='Compare equal-sized before and after images')
    c.add_argument('before', type=Path)
    c.add_argument('after', type=Path)
    c.add_argument('--out', default='reports/custom')
    c.add_argument('--threshold', type=float, default=36)
    c.add_argument('--min-area', type=int, default=50)
    args = parser.parse_args()
    try:
        if args.command == 'benchmark':
            from .benchmark import run
            run(args.out)
            return 0
        config = Config(threshold=args.threshold, min_area=args.min_area)
        before, after = load_pair(args.before, args.after)
        result = inspect(before, after, config)
        payload = dict(title='Visual Change Inspector', mode='custom', config=asdict(config),
                       summary=result.summary(), examples=[example('Your image comparison', before, after, result)],
                       processed_dimensions=[before.shape[1], before.shape[0]])
        write_report(args.out, payload)
        if result.mask is not None:
            Image.fromarray(result.mask.astype(np.uint8)*255).save(Path(args.out)/'mask.png')
            Image.fromarray(result.valid.astype(np.uint8)*255).save(Path(args.out)/'valid-area.png')
            Image.fromarray(result.aligned).save(Path(args.out)/'aligned.png')
        else:
            # Reusing an output directory must not leave a previous detection behind.
            for name in ('mask.png', 'valid-area.png', 'aligned.png'):
                (Path(args.out)/name).unlink(missing_ok=True)
        print(f'{result.status}: {result.reason}\nReport: {Path(args.out).resolve()/"report.html"}')
        return 2 if result.status == 'REVIEW' else 0
    except (ValueError, OSError) as exc:
        print(f'Input error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
