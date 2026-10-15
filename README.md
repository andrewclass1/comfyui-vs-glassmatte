# comfyui-vs-glassmatte

ComfyUI nodes by View Source for making clean, 16-bit RGBA hero images of glass bottles, with true partial transparency in the glass. Torch and numpy only, no extra pip installs.

## Install

ComfyUI-Manager → **Install via Git URL** → `https://github.com/<owner>/comfyui-vs-glassmatte`, then restart.
Or clone into `ComfyUI/custom_nodes/`.

## Nodes

| Node | What it does |
|---|---|
| VS Pad + Flatten | Flattens an RGBA render on white or black and pads it to a fixed aspect ratio |
| VS Label Photo Slot | Drops a photo into a label window, or marks areas to keep from the base image |
| VS Cylinder Label Wrap | Wraps flat label art around a cylindrical bottle body |
| VS Label Shade + Composite | Lights the clean label with the base image's per-column lighting and composites it |
| VS Align To Reference | Puts a generated image back in the source frame using the bottle silhouette |
| VS Normalize White Plate | Divides out the backdrop colour and forces the background to exactly 1.0 |
| VS Glass Band Mask | Finds the empty glass neck between cap and liquid line |
| VS Black Plate From White | Builds the black plate from the white plate with the bottle mask |
| VS Mask Crop / VS Mask Stitch | Crop a masked region for inpainting and stitch it back |
| VS Plate Drift Check | Confirms the two plates line up (PASS / FLAG) |
| VS Difference Matte | `alpha = 1 - (W - B)`, straight colour `= B / alpha` |
| VS Composite Over | Previews the result over red, grey, checker or an image |
| VS Save RGBA (16-bit) | Writes 16-bit or 8-bit straight-alpha PNGs |

## Pipeline

1. Flatten the source on white and lock the clean label onto it.
2. Refine with an image model on a pure white backdrop.
3. Align, normalise the backdrop, lock the label again: the **white plate**.
4. Remove the backdrop with the bottle mask, then inpaint only the glass neck on black: the **black plate**.
5. Difference matte to 16-bit RGBA.

Because the black plate is built from the white plate, opaque areas are identical in both by construction. Only the glass is regenerated.

## Calibration

Wrap defaults are x as a fraction of canvas width and y as a fraction of canvas height, measured on a 9:16 canvas. Label rects are fractions of the flat label. Refit `center_x`, `radius`, `top`, `bottom` and `u_center` for a different bottle.

## Reports

| Report | Healthy |
|---|---|
| Align | `status: OK` (bottle width within 1.5% of the reference) |
| Glass band | `mode_used: auto` |
| Drift | `status: PASS` |
