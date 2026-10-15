"""
View Source Glass Matte nodes for ComfyUI.

Built for a glass bottle hero pipeline:
  source render -> label pre-lock -> Nano Banana Pro -> align -> label lock
  -> white plate -> black plate (Flux Fill on the glass neck only)
  -> difference matte -> 16-bit RGBA.

Only torch + numpy are required. No OpenCV, no SciPy.
All IMAGE tensors are [B, H, W, C] float 0..1, MASK tensors are [B, H, W].
Nodes process the first item of a batch; the pipeline is built for single heroes.
"""

import json
import math
import os
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F

try:  # available inside ComfyUI
    import folder_paths
except ImportError:  # unit tests outside ComfyUI
    folder_paths = None

CAT = "View Source/Glass Matte"

# ---------------------------------------------------------------------------
# Default calibration: a 12 oz crown-cap bottle render padded to 9:16
# (1487 x 2644 canvas). x values are fractions of canvas width, y values are
# fractions of canvas height. Label rects are fractions of the flat label.
# Any render sharing that bottle and label die line can reuse these.
# ---------------------------------------------------------------------------
WRAP = dict(center_x=0.49896, radius=0.21324, top=0.51365, bottom=0.85307,
            u_center=0.27956, u_scale=1.0)
PHOTO_RECT = (0.1367, 0.2751, 0.4250, 0.7799)   # u0, v0, u1, v1
NAME_RECT = (0.1041, 0.5816, 0.1359, 0.7821)
GLASS_BAND = (0.043, 0.185)                      # fractions of bottle height
LUMA = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _img(t):
    """IMAGE [B,H,W,C] -> [H,W,C] float32."""
    return t[0].float()


def _msk(t, h=None, w=None):
    """MASK [B,H,W] or [H,W] -> [H,W] float32, resized if needed."""
    m = t[0] if t.dim() == 3 else t
    m = m.float()
    if h is not None and (m.shape[0] != h or m.shape[1] != w):
        m = F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return m.clamp(0, 1)


def _out_img(x):
    return x.clamp(0, 1)[None].contiguous()


def _out_msk(x):
    return x.clamp(0, 1)[None].contiguous()


def _lum(img):
    w = torch.tensor(LUMA, dtype=img.dtype, device=img.device)
    return (img[..., :3] * w).sum(-1)


def _resize_img(img, h, w):
    """[H,W,C] -> [h,w,C] with antialiasing when shrinking."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    shrink = h < img.shape[0] or w < img.shape[1]
    x = img.permute(2, 0, 1)[None]
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=shrink)
    return x[0].permute(1, 2, 0)


def _resize_msk(m, h, w):
    return _resize_img(m[..., None], h, w)[..., 0]


def _gauss_kernel(sigma, device):
    sigma = max(float(sigma), 1e-3)
    r = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum(), r


def _blur1d_last(x, sigma):
    """Gaussian blur along the last dim of a 1D or 2D tensor (reflect padded)."""
    if sigma <= 0:
        return x
    k, r = _gauss_kernel(sigma, x.device)
    shp = x.shape
    y = x.reshape(-1, 1, shp[-1])
    y = F.pad(y, (r, r), mode="replicate")
    y = F.conv1d(y, k[None, None])
    return y.reshape(shp)


def _blur2d(m, sigma):
    if sigma <= 0:
        return m
    y = _blur1d_last(m, sigma)
    y = _blur1d_last(y.transpose(-1, -2), sigma).transpose(-1, -2)
    return y


def _dilate(m, px):
    if px <= 0:
        return m
    k = 2 * int(px) + 1
    return F.max_pool2d(m[None, None], k, stride=1, padding=int(px))[0, 0]


def _erode(m, px):
    if px <= 0:
        return m
    return 1 - _dilate(1 - m, px)


def _extent(m, thr=0.5, min_px=2):
    """Row/col extent of a mask, robust to stray pixels."""
    b = m > thr
    rows = torch.nonzero(b.sum(1) >= min_px).flatten()
    cols = torch.nonzero(b.sum(0) >= min_px).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        raise ValueError("Mask is empty; check the BiRefNet / alpha input.")
    y0, y1 = rows[0].item(), rows[-1].item() + 1
    x0, x1 = cols[0].item(), cols[-1].item() + 1
    w = m * b
    tot = w.sum()
    ys = torch.arange(m.shape[0], device=m.device, dtype=torch.float32)
    xs = torch.arange(m.shape[1], device=m.device, dtype=torch.float32)
    cy = (w.sum(1) * ys).sum() / tot
    cx = (w.sum(0) * xs).sum() / tot
    return dict(y0=y0, y1=y1, x0=x0, x1=x1, cx=cx.item(), cy=cy.item())


def _parse_aspect(s):
    a, b = s.split(":")
    return float(a) / float(b)


# ---------------------------------------------------------------------------
# 16-bit PNG writer (PIL cannot write 16-bit RGBA)
# ---------------------------------------------------------------------------
def write_png(path, arr):
    """arr: HxWx4 (or x3) uint8 or uint16 numpy array."""
    arr = np.ascontiguousarray(arr)
    h, w, c = arr.shape
    bit = 16 if arr.dtype == np.uint16 else 8
    ctype = {3: 2, 4: 6}[c]
    data = arr.astype(">u2" if bit == 16 else np.uint8).view(np.uint8).reshape(h, -1)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), data], 1).tobytes()

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bit, ctype, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# 1. Pad + flatten the source render
# ---------------------------------------------------------------------------
class VSPadFlatten:
    """Flattens an RGBA render onto a solid background and pads it to an
    aspect ratio Nano Banana Pro supports, so framing stays fixed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "aspect": (["9:16", "2:3", "3:4", "4:5", "1:1", "source"], {"default": "9:16"}),
            "background": (["white", "black"], {"default": "white"}),
            "mask_is_loadimage": ("BOOLEAN", {"default": True,
                                  "tooltip": "LoadImage outputs 1 = transparent. Leave on when the mask comes from LoadImage."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "bottle_mask")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, mask, aspect, background, mask_is_loadimage):
        img = _img(image)[..., :3]
        h, w = img.shape[:2]
        m = mask[0] if mask.dim() == 3 else mask
        if m.shape[0] != h or m.shape[1] != w:      # LoadImage gives 64x64 zeros when there is no alpha
            a = torch.ones(h, w, device=img.device)
        else:
            a = (1 - m.float()) if mask_is_loadimage else m.float()
        bg = 1.0 if background == "white" else 0.0
        flat = img * a[..., None] + bg * (1 - a[..., None])

        if aspect == "source":
            return (_out_img(flat), _out_msk(a))
        r = _parse_aspect(aspect)
        tw, th = w, h
        if w / h < r:
            tw = int(round(h * r))
        else:
            th = int(round(w / r))
        canvas = torch.full((th, tw, 3), bg, device=img.device)
        amask = torch.zeros(th, tw, device=img.device)
        ox, oy = (tw - w) // 2, (th - h) // 2
        canvas[oy:oy + h, ox:ox + w] = flat
        amask[oy:oy + h, ox:ox + w] = a
        return (_out_img(canvas), _out_msk(amask))


# ---------------------------------------------------------------------------
# 2. Align a generated image back onto the reference silhouette
# ---------------------------------------------------------------------------
class VSAlignToReference:
    """Puts the Nano Banana output back into the source frame: uniform scale
    from bottle height, translation from bottle centre. Everything downstream
    (label wrap, glass band) is calibrated in that frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "image_mask": ("MASK",),
            "reference_mask": ("MASK",),
            "output_height": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 16}),
            "fill": (["white", "black"], {"default": "white"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "report")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, image_mask, reference_mask, output_height, fill):
        img = _img(image)[..., :3]
        hm, wm = img.shape[:2]
        mm = _msk(image_mask, hm, wm)
        rm = _msk(reference_mask)
        hr, wr = rm.shape
        Hc = int(output_height)
        Wc = int(round(Hc * wr / hr))
        Wc -= Wc % 2

        re, me = _extent(rm), _extent(mm)
        ry0, ry1 = re["y0"] / hr, re["y1"] / hr
        rcx = (re["x0"] + re["x1"]) / 2 / wr
        rw = (re["x1"] - re["x0"]) / wr * Wc
        mcx = (me["x0"] + me["x1"]) / 2
        s = (me["y1"] - me["y0"]) / ((ry1 - ry0) * Hc)        # moving px per canvas px

        dev = img.device
        Y = torch.arange(Hc, device=dev, dtype=torch.float32) + 0.5
        X = torch.arange(Wc, device=dev, dtype=torch.float32) + 0.5
        ym = me["y0"] + (Y - ry0 * Hc) * s
        xm = mcx + (X - rcx * Wc) * s
        gy = ym / hm * 2 - 1
        gx = xm / wm * 2 - 1
        grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]
        stack = torch.cat([img, mm[..., None], torch.ones_like(mm)[..., None]], -1).permute(2, 0, 1)[None]
        out = F.grid_sample(stack, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0].permute(1, 2, 0)
        valid = out[..., 4:5]
        bg = 1.0 if fill == "white" else 0.0
        rgb = out[..., :3] + bg * (1 - valid)
        mask = out[..., 3]

        width_ratio = ((me["x1"] - me["x0"]) / s) / max(rw, 1)
        rep = {
            "scale_src_px_per_out_px": round(s, 4),
            "out_size": [Wc, Hc],
            "bottle_width_vs_reference": round(width_ratio, 4),
            "status": "OK" if abs(width_ratio - 1) < 0.015 else
                      "CHECK: bottle proportions changed by more than 1.5%, label fit may drift",
        }
        return (_out_img(rgb), _out_msk(mask), json.dumps(rep, indent=2))


# ---------------------------------------------------------------------------
# 3. Photo slot / keep areas on the flat label
# ---------------------------------------------------------------------------
class VSLabelPhotoSlot:
    """The EPS exports have an empty photo window and blank NAME/FROM fields.
    Either drop the bottle's photo in, or mark those areas as 'keep from base'
    so the photo already on the bottle survives the label lock."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "label": ("IMAGE",),
            "photo_u0": ("FLOAT", {"default": PHOTO_RECT[0], "min": 0, "max": 1, "step": 0.0001}),
            "photo_v0": ("FLOAT", {"default": PHOTO_RECT[1], "min": 0, "max": 1, "step": 0.0001}),
            "photo_u1": ("FLOAT", {"default": PHOTO_RECT[2], "min": 0, "max": 1, "step": 0.0001}),
            "photo_v1": ("FLOAT", {"default": PHOTO_RECT[3], "min": 0, "max": 1, "step": 0.0001}),
            "keep_name_fields": ("BOOLEAN", {"default": True}),
            "name_u0": ("FLOAT", {"default": NAME_RECT[0], "min": 0, "max": 1, "step": 0.0001}),
            "name_v0": ("FLOAT", {"default": NAME_RECT[1], "min": 0, "max": 1, "step": 0.0001}),
            "name_u1": ("FLOAT", {"default": NAME_RECT[2], "min": 0, "max": 1, "step": 0.0001}),
            "name_v1": ("FLOAT", {"default": NAME_RECT[3], "min": 0, "max": 1, "step": 0.0001}),
        }, "optional": {
            "photo": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("label", "coverage")
    FUNCTION = "run"
    CATEGORY = CAT

    @staticmethod
    def _box(h, w, u0, v0, u1, v1):
        return (int(round(v0 * h)), int(round(v1 * h)), int(round(u0 * w)), int(round(u1 * w)))

    def run(self, label, photo_u0, photo_v0, photo_u1, photo_v1, keep_name_fields,
            name_u0, name_v0, name_u1, name_v1, photo=None):
        lab = _img(label)[..., :3].clone()
        h, w = lab.shape[:2]
        cov = torch.ones(h, w, device=lab.device)
        y0, y1, x0, x1 = self._box(h, w, photo_u0, photo_v0, photo_u1, photo_v1)
        if photo is None:
            cov[y0:y1, x0:x1] = 0
        else:
            ph = _img(photo)[..., :3]
            bh, bw = y1 - y0, x1 - x0
            sc = max(bh / ph.shape[0], bw / ph.shape[1])          # cover fit
            rh, rw = int(math.ceil(ph.shape[0] * sc)), int(math.ceil(ph.shape[1] * sc))
            ph = _resize_img(ph, rh, rw)
            oy, ox = (rh - bh) // 2, (rw - bw) // 2
            lab[y0:y1, x0:x1] = ph[oy:oy + bh, ox:ox + bw]
        if keep_name_fields:
            a0, a1, b0, b1 = self._box(h, w, name_u0, name_v0, name_u1, name_v1)
            cov[a0:a1, b0:b1] = 0
        return (_out_img(lab), _out_msk(cov))


# ---------------------------------------------------------------------------
# 4. Cylindrical label wrap
# ---------------------------------------------------------------------------
class VSCylinderLabelWrap:
    """Wraps the flat label around the bottle body (orthographic cylinder).
    Defaults are the reference bottle calibration; they hold for every variant because
    the bottle and label die line are shared."""

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo=-1.0, hi=2.0, st=0.00001: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st})
        return {"required": {
            "label": ("IMAGE",),
            "coverage": ("MASK",),
            "reference": ("IMAGE",),
            "center_x": f(WRAP["center_x"]),
            "radius": f(WRAP["radius"], 0.01, 1.0),
            "top": f(WRAP["top"]),
            "bottom": f(WRAP["bottom"]),
            "u_center": f(WRAP["u_center"]),
            "u_scale": f(WRAP["u_scale"], 0.5, 2.0),
            "edge_fade_start": ("FLOAT", {"default": 0.93, "min": 0.5, "max": 1.0, "step": 0.005,
                                "tooltip": "Fraction of the radius where the label starts handing over to the base image edge."}),
            "edge_fade_end": ("FLOAT", {"default": 0.975, "min": 0.5, "max": 1.0, "step": 0.005}),
            "supersample": ("INT", {"default": 2, "min": 1, "max": 3}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("label", "mask")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, label, coverage, reference, center_x, radius, top, bottom, u_center, u_scale,
            edge_fade_start, edge_fade_end, supersample):
        ref = _img(reference)
        Hc, Wc = ref.shape[:2]
        dev = ref.device
        lab = _img(label)[..., :3].to(dev)
        cov = _msk(coverage, lab.shape[0], lab.shape[1]).to(dev)

        ss = int(supersample)
        H, W = Hc * ss, Wc * ss
        # Pre-shrink the label to ~1 label px per output px at the front of the bottle,
        # so text is antialiased before the warp.
        lab_h = max(16, int(round((bottom - top) * H)))
        px_per_u = 2 * math.pi * radius * W / max(u_scale, 1e-6)
        lab_w = max(16, int(round(px_per_u)))
        if lab_h < lab.shape[0]:
            lab = _resize_img(lab, lab_h, lab_w)
            cov = _resize_msk(cov, lab_h, lab_w)

        Y = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5)
        X = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5)
        s = (X - center_x * W) / (radius * W)
        u = u_center + u_scale * torch.asin(s.clamp(-1, 1)) / (2 * math.pi)
        u = torch.remainder(u, 1.0)
        v = (Y - top * H) / ((bottom - top) * H)
        gx = (u * 2 - 1)[None, :].expand(H, W)
        gy = (v * 2 - 1)[:, None].expand(H, W)
        grid = torch.stack([gx, gy], -1)[None]
        src = torch.cat([lab, cov[..., None]], -1).permute(2, 0, 1)[None]
        out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]

        # vertical extent with 1px antialias, horizontal hand-over near the silhouette
        dy_top = Y - top * H
        dy_bot = bottom * H - Y
        vmask = (dy_top.clamp(0, 1) * dy_bot.clamp(0, 1))[:, None]
        sa = s.abs()
        hfade = 1 - ((sa - edge_fade_start) / max(edge_fade_end - edge_fade_start, 1e-4)).clamp(0, 1)
        hfade = hfade * hfade * (3 - 2 * hfade)
        mask = out[3] * vmask * hfade[None, :]
        rgb = out[:3].permute(1, 2, 0)

        if ss > 1:
            rgb = _resize_img(rgb, Hc, Wc)
            mask = _resize_msk(mask, Hc, Wc)
        return (_out_img(rgb), _out_msk(mask))


# ---------------------------------------------------------------------------
# 5. Shade the clean label with the base image's lighting and composite
# ---------------------------------------------------------------------------
class VSLabelShadeComposite:
    """Fits base = label * gain(x) + lift(x) per column (smoothed across x)
    and applies it to the clean label. Cylinder lighting and the vertical
    specular streaks are captured; the label pixels stay the source art."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("IMAGE",),
            "label": ("IMAGE",),
            "mask": ("MASK",),
            "column_smoothing_px": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 200.0, "step": 0.5}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "regularize": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                           "tooltip": "Pulls gain toward 1 on columns with little label contrast."}),
        }, "optional": {
            "clip_to_mask": ("MASK", {"tooltip": "Bottle mask. Keeps label pixels inside the silhouette."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "shaded_label")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, base, label, mask, column_smoothing_px, strength, regularize, clip_to_mask=None):
        b = _img(base)[..., :3]
        H, W = b.shape[:2]
        lab = _img(label)[..., :3].to(b.device)
        if lab.shape[0] != H or lab.shape[1] != W:
            lab = _resize_img(lab, H, W)
        m = _msk(mask, H, W).to(b.device)
        if clip_to_mask is not None:
            m = m * _erode((_msk(clip_to_mask, H, W).to(b.device) > 0.5).float(), 1)

        x = _lum(lab)
        y = _lum(b)
        wt = (m > 0.95).float()
        sw = wt.sum(0)
        sx, sy = (wt * x).sum(0), (wt * y).sum(0)
        sxx, sxy = (wt * x * x).sum(0), (wt * x * y).sum(0)
        sig = column_smoothing_px
        sw, sx, sy, sxx, sxy = [_blur1d_last(t, sig) for t in (sw, sx, sy, sxx, sxy)]
        n = sw.clamp_min(1e-3)
        mx, my = sx / n, sy / n
        var = sxx / n - mx * mx
        cov = sxy / n - mx * my
        gain = (cov + regularize) / (var + regularize)
        gain = gain.clamp(0.05, 3.0)
        lift = (my - gain * mx).clamp(-0.5, 0.9)

        shaded = (lab * gain[None, :, None] + lift[None, :, None]).clamp(0, 1)
        shaded = lab + (shaded - lab) * strength
        out = b * (1 - m[..., None]) + shaded * m[..., None]
        return (_out_img(out), _out_img(shaded))


# ---------------------------------------------------------------------------
# 6. White plate normalisation
# ---------------------------------------------------------------------------
class VSNormalizeWhitePlate:
    """Generated 'white' backgrounds are rarely 1.0 and often carry a soft
    shadow. Difference matting needs a true 1.0 background, so this measures
    the backdrop, divides it out and forces everything outside the bottle to 1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "bottle_mask": ("MASK",),
            "outside_threshold": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005}),
            "sample_margin_px": ("INT", {"default": 24, "min": 0, "max": 512}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, bottle_mask, outside_threshold, sample_margin_px):
        img = _img(image)[..., :3]
        H, W = img.shape[:2]
        m = _msk(bottle_mask, H, W).to(img.device)
        far = _dilate((m > outside_threshold).float(), sample_margin_px) < 0.5
        if far.sum() < 100:
            raise ValueError("No background pixels to sample; lower sample_margin_px.")
        bg = img[far].median(0).values.clamp_min(0.5)
        out = (img / bg).clamp(0, 1)
        out[m <= outside_threshold] = 1.0
        rep = {"measured_background_rgb": [round(v, 4) for v in bg.tolist()]}
        return (_out_img(out), json.dumps(rep))


# ---------------------------------------------------------------------------
# 7. Glass band mask (empty neck between cap and liquid line)
# ---------------------------------------------------------------------------
class VSGlassBandMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "bottle_mask": ("MASK",),
            "mode": (["auto", "manual"], {"default": "auto"}),
            "top_frac": ("FLOAT", {"default": GLASS_BAND[0], "min": 0, "max": 1, "step": 0.001}),
            "bottom_frac": ("FLOAT", {"default": GLASS_BAND[1], "min": 0, "max": 1, "step": 0.001}),
            "brightness_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            "search_limit_frac": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.01}),
            "grow_px": ("INT", {"default": 6, "min": 0, "max": 128}),
        }}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("glass_mask", "report")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, bottle_mask, mode, top_frac, bottom_frac, brightness_threshold,
            search_limit_frac, grow_px):
        img = _img(image)[..., :3]
        H, W = img.shape[:2]
        m = _msk(bottle_mask, H, W).to(img.device)
        e = _extent(m)
        bh = e["y1"] - e["y0"]
        y_top = e["y0"] + int(top_frac * bh)
        y_bot = e["y0"] + int(bottom_frac * bh)
        used = "manual"
        if mode == "auto":
            L = _lum(img)
            lim = e["y0"] + int(search_limit_frac * bh)
            bright = []
            for yy in range(e["y0"], lim):
                cols = torch.nonzero(m[yy] > 0.5).flatten()
                if cols.numel() < 4:
                    bright.append(False)
                    continue
                c0, c1 = cols[0].item(), cols[-1].item()
                q = (c1 - c0) // 5
                bright.append(bool(L[yy, c0 + 2 * q:c1 - 2 * q + 1].mean() > brightness_threshold))
            # cap = dark rows at the top; band runs from the first bright row
            # after the cap to the first sustained dark run (the liquid).
            # Thin dark lines inside the neck (ring shadows) are tolerated.
            min_run = max(3, int(bh * 0.015))
            first_bright, run = None, 0
            for i, f in enumerate(bright):          # skip the cap (and its highlights)
                run = 0 if f else run + 1
                if run >= min_run:
                    first_bright = next((j for j in range(i, len(bright)) if bright[j]), None)
                    break
            liquid = None
            if first_bright is not None:
                run = 0
                for i in range(first_bright, len(bright)):
                    run = 0 if bright[i] else run + 1
                    if run >= min_run:
                        liquid = i - run + 1
                        break
            if first_bright is not None and liquid is not None and liquid - first_bright > bh * 0.03:
                y_top, y_bot = e["y0"] + first_bright, e["y0"] + liquid
                used = "auto"
        band = torch.zeros_like(m)
        band[y_top:y_bot] = (m[y_top:y_bot] > 0.05).float()
        band = _dilate(band, grow_px)
        band[:y_top] = 0          # never grow up into the cap
        rep = {"mode_used": used, "rows": [y_top, y_bot],
               "as_fraction_of_bottle": [round((y_top - e["y0"]) / bh, 4), round((y_bot - e["y0"]) / bh, 4)]}
        return (_out_msk(band), json.dumps(rep))


# ---------------------------------------------------------------------------
# 8. Black plate base, crop / stitch for Flux Fill
# ---------------------------------------------------------------------------
class VSBlackPlateFromWhite:
    """Deterministic black plate: removes the white backdrop contribution
    using the bottle mask. Opaque bottle pixels are copied unchanged, so the
    plates cannot drift there. The glass band is replaced by Flux Fill next."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "white_plate": ("IMAGE",),
            "bottle_mask": ("MASK",),
            "solid_above": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.005,
                            "tooltip": "Mask values above this count as fully opaque, so the body never goes see-through."}),
            "clear_below": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, white_plate, bottle_mask, solid_above, clear_below):
        w = _img(white_plate)[..., :3]
        m = _msk(bottle_mask, w.shape[0], w.shape[1]).to(w.device)
        m = ((m - clear_below) / max(solid_above - clear_below, 1e-4)).clamp(0, 1)
        return (_out_img((w - (1 - m[..., None])).clamp(0, 1)),)


class VSMaskCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "padding_px": ("INT", {"default": 96, "min": 0, "max": 2048}),
            "target_long_side": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "VS_CROP")
    RETURN_NAMES = ("image", "mask", "crop")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, mask, padding_px, target_long_side):
        img = _img(image)[..., :3]
        H, W = img.shape[:2]
        m = _msk(mask, H, W).to(img.device)
        e = _extent(m, thr=0.01, min_px=1)
        y0, y1 = max(0, e["y0"] - padding_px), min(H, e["y1"] + padding_px)
        x0, x1 = max(0, e["x0"] - padding_px), min(W, e["x1"] + padding_px)
        ci, cm = img[y0:y1, x0:x1], m[y0:y1, x0:x1]
        ch, cw = ci.shape[:2]
        sc = target_long_side / max(ch, cw)
        th = max(16, int(round(ch * sc / 16)) * 16)
        tw = max(16, int(round(cw * sc / 16)) * 16)
        crop = {"box": [y0, y1, x0, x1], "size": [H, W]}
        return (_out_img(_resize_img(ci, th, tw)), _out_msk(_resize_msk(cm, th, tw)), crop)


class VSMaskStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("IMAGE",),
            "patch": ("IMAGE",),
            "crop": ("VS_CROP",),
            "mask": ("MASK",),
            "feather_px": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 64.0, "step": 0.5}),
        }, "optional": {
            "force_black_outside": ("MASK", {"tooltip": "Bottle mask. Pixels outside it are set to exact black after stitching."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, base, patch, crop, mask, feather_px, force_black_outside=None):
        b = _img(base)[..., :3].clone()
        H, W = b.shape[:2]
        y0, y1, x0, x1 = crop["box"]
        p = _resize_img(_img(patch)[..., :3].to(b.device), y1 - y0, x1 - x0)
        m = _msk(mask, H, W).to(b.device)
        m = _blur2d(m, feather_px)[y0:y1, x0:x1, None]
        b[y0:y1, x0:x1] = b[y0:y1, x0:x1] * (1 - m) + p * m
        if force_black_outside is not None:
            bm = _msk(force_black_outside, H, W).to(b.device)
            b = b * (bm > 0.004).float()[..., None]
        return (_out_img(b),)


# ---------------------------------------------------------------------------
# 9. Drift check
# ---------------------------------------------------------------------------
def _phase_shift(a, b):
    A, B = torch.fft.rfft2(a - a.mean()), torch.fft.rfft2(b - b.mean())
    R = A * B.conj()
    R = R / R.abs().clamp_min(1e-8)
    r = torch.fft.irfft2(R, s=a.shape)
    idx = torch.argmax(r)
    dy, dx = divmod(idx.item(), a.shape[1])
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return dy, dx


class VSPlateDriftCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "white_plate": ("IMAGE",),
            "black_plate": ("IMAGE",),
            "bottle_mask": ("MASK",),
            "glass_mask": ("MASK",),
            "max_shift_px": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("report", "difference_heatmap")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, white_plate, black_plate, bottle_mask, glass_mask, max_shift_px):
        w = _img(white_plate)[..., :3]
        b = _img(black_plate)[..., :3].to(w.device)
        H, Wd = w.shape[:2]
        bm = _msk(bottle_mask, H, Wd).to(w.device)
        gm = _msk(glass_mask, H, Wd).to(w.device)
        opaque = (bm > 0.98) & (_dilate((gm > 0.01).float(), 8) < 0.5)
        d = (w - b).abs().mean(-1)
        od = d[opaque]
        e = _extent((gm > 0.01).float(), 0.5, 1)
        be = _extent(bm)
        y0, y1 = max(0, be["y0"] - 32), min(H, e["y1"] + 64)
        x0, x1 = max(0, e["x0"] - 64), min(Wd, e["x1"] + 64)
        sil_w = (_lum(w) < 0.995).float()[y0:y1, x0:x1]
        sil_b = (_lum(b) > 0.005).float()[y0:y1, x0:x1]
        dy, dx = _phase_shift(sil_w, sil_b)
        shift = math.hypot(dy, dx)
        ok = shift <= max_shift_px and od.mean().item() < 0.002
        rep = {
            "status": "PASS" if ok else "FLAG",
            "opaque_region_mean_abs_diff": round(od.mean().item(), 6),
            "opaque_region_max_abs_diff": round(od.max().item(), 6),
            "neck_silhouette_shift_px": [dy, dx],
        }
        heat = torch.stack([d * 4 * opaque, d * gm, torch.zeros_like(d)], -1).clamp(0, 1)
        return (json.dumps(rep, indent=2), _out_img(heat))


# ---------------------------------------------------------------------------
# 10. Difference matte
# ---------------------------------------------------------------------------
class VSDifferenceMatte:
    """alpha = 1 - (W - B) / (white_level - black_level)
    colour = (B - (1 - alpha) * black_level) / alpha   (straight, unpremultiplied)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "white_plate": ("IMAGE",),
            "black_plate": ("IMAGE",),
            "white_level": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.0, "step": 0.001}),
            "black_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.001}),
            "clip_low": ("FLOAT", {"default": 0.004, "min": 0.0, "max": 0.2, "step": 0.001}),
            "clip_high": ("FLOAT", {"default": 0.996, "min": 0.8, "max": 1.0, "step": 0.001}),
            "channel_combine": (["mean", "max", "luma"], {"default": "mean"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("color", "alpha", "premultiplied")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, white_plate, black_plate, white_level, black_level, clip_low, clip_high, channel_combine):
        w = _img(white_plate)[..., :3]
        b = _img(black_plate)[..., :3].to(w.device)
        if b.shape != w.shape:
            raise ValueError("White and black plates must be the same size.")
        ac = 1 - (w - b) / (white_level - black_level)
        if channel_combine == "max":
            a = ac.max(-1).values
        elif channel_combine == "luma":
            a = _lum(ac)
        else:
            a = ac.mean(-1)
        a = ((a - clip_low) / (clip_high - clip_low)).clamp(0, 1)
        col = (b - (1 - a[..., None]) * black_level) / a[..., None].clamp_min(1e-4)
        col = torch.where(a[..., None] > 0, col, torch.zeros_like(col)).clamp(0, 1)
        return (_out_img(col), _out_msk(a), _out_img(col * a[..., None]))


# ---------------------------------------------------------------------------
# 11. Previews and save
# ---------------------------------------------------------------------------
class VSCompositeOver:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "color": ("IMAGE",),
            "alpha": ("MASK",),
            "background": (["red", "mid grey", "checker", "black", "white"], {"default": "red"}),
        }, "optional": {"background_image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, color, alpha, background, background_image=None):
        c = _img(color)[..., :3]
        H, W = c.shape[:2]
        a = _msk(alpha, H, W).to(c.device)[..., None]
        if background_image is not None:
            bg = _img(background_image)[..., :3].to(c.device)
            sc = max(H / bg.shape[0], W / bg.shape[1])
            rh, rw = int(math.ceil(bg.shape[0] * sc)), int(math.ceil(bg.shape[1] * sc))
            bg = _resize_img(bg, rh, rw)
            oy, ox = (rh - H) // 2, (rw - W) // 2
            bg = bg[oy:oy + H, ox:ox + W]
        elif background == "checker":
            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            chk = (((yy // 64) + (xx // 64)) % 2).float().to(c.device) * 0.25 + 0.6
            bg = chk[..., None].expand(H, W, 3)
        else:
            rgb = {"red": (1, 0, 0), "mid grey": (0.5, 0.5, 0.5), "black": (0, 0, 0), "white": (1, 1, 1)}[background]
            bg = torch.tensor(rgb, dtype=c.dtype, device=c.device).expand(H, W, 3)
        return (_out_img(c * a + bg * (1 - a)),)


class VSSaveRGBA16:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "color": ("IMAGE",),
            "alpha": ("MASK",),
            "filename_prefix": ("STRING", {"default": "hero/hero_RGBA"}),
            "bit_depth": (["16", "8"], {"default": "16"}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, color, alpha, filename_prefix, bit_depth):
        c = _img(color)[..., :3]
        H, W = c.shape[:2]
        a = _msk(alpha, H, W).to(c.device)
        rgba = torch.cat([c, a[..., None]], -1).clamp(0, 1).cpu().numpy()
        if folder_paths is not None:
            out_dir = folder_paths.get_output_directory()
            full, name, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, W, H)
        else:
            full, name, counter, sub = os.getcwd(), os.path.basename(filename_prefix), 1, ""
        fname = f"{name}_{counter:05}_.png"
        path = os.path.join(full, fname)
        if bit_depth == "16":
            write_png(path, np.round(rgba * 65535).astype(np.uint16))
        else:
            write_png(path, np.round(rgba * 255).astype(np.uint8))

        ui = {}
        if folder_paths is not None:   # 8-bit preview for the UI
            tdir = folder_paths.get_temp_directory()
            tname = f"vs_rgba_preview_{counter:05}.png"
            write_png(os.path.join(tdir, tname), np.round(rgba * 255).astype(np.uint8))
            ui = {"images": [{"filename": tname, "subfolder": "", "type": "temp"}]}
        print(f"[VS Glass Matte] saved {path}")
        return {"ui": ui}


NODE_CLASS_MAPPINGS = {
    "VSPadFlatten": VSPadFlatten,
    "VSAlignToReference": VSAlignToReference,
    "VSLabelPhotoSlot": VSLabelPhotoSlot,
    "VSCylinderLabelWrap": VSCylinderLabelWrap,
    "VSLabelShadeComposite": VSLabelShadeComposite,
    "VSNormalizeWhitePlate": VSNormalizeWhitePlate,
    "VSGlassBandMask": VSGlassBandMask,
    "VSBlackPlateFromWhite": VSBlackPlateFromWhite,
    "VSMaskCrop": VSMaskCrop,
    "VSMaskStitch": VSMaskStitch,
    "VSPlateDriftCheck": VSPlateDriftCheck,
    "VSDifferenceMatte": VSDifferenceMatte,
    "VSCompositeOver": VSCompositeOver,
    "VSSaveRGBA16": VSSaveRGBA16,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VSPadFlatten": "VS Pad + Flatten",
    "VSAlignToReference": "VS Align To Reference",
    "VSLabelPhotoSlot": "VS Label Photo Slot",
    "VSCylinderLabelWrap": "VS Cylinder Label Wrap",
    "VSLabelShadeComposite": "VS Label Shade + Composite",
    "VSNormalizeWhitePlate": "VS Normalize White Plate",
    "VSGlassBandMask": "VS Glass Band Mask",
    "VSBlackPlateFromWhite": "VS Black Plate From White",
    "VSMaskCrop": "VS Mask Crop",
    "VSMaskStitch": "VS Mask Stitch",
    "VSPlateDriftCheck": "VS Plate Drift Check",
    "VSDifferenceMatte": "VS Difference Matte",
    "VSCompositeOver": "VS Composite Over",
    "VSSaveRGBA16": "VS Save RGBA (16-bit)",
}
