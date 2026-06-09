# Cell Genesis Studio

This package provides a local Python desktop app for CellUniverse manual initial cell setup, 3D review, and video export.

Main goals:

- Load CellUniverse `cells.csv`, real tif, synth tif, and fixed GT in one place.
- Show prediction centers, GT centers, and ellipsoid wire outlines with a small number of Napari layers.
- Keep all prediction outlines in one Napari Shapes layer with per shape colors, instead of many color layers.
- Export separate overlay files only for quick 2D or slice review without changing the original algorithm outputs.
- Provide a base for MP4 and GIF export.

Current validation policy:

- Use `/Users/wangyiding/CellUniverse/C++/config/embryo/ground_truth/embryo_FixedGroundTruth.csv` for review.
- Keep `/Users/wangyiding/CellUniverse/C++/config/embryo/ground_truth/embryo_FixedGroundTruth_changes.csv` as the delayed split correction record.
- Do not mark frame 173 as verified until the suspected delayed split case is visually checked.

Example commands:

Open the local PyQt app for manual initial cell setup and video export:

```bash
cd /Users/wangyiding/CellUniverse/Cell_Genesis_Studio
PYTHONPATH=src python3 -m celluniverse_napari_app.local_app
```

Build a clickable macOS app bundle:

```bash
cd /Users/wangyiding/CellUniverse/Cell_Genesis_Studio
chmod +x scripts/build_macos_app.sh
scripts/build_macos_app.sh
open "Dist/Cell Genesis Studio.app"
```

If the app should always use a specific Python interpreter, set `CELLUNIVERSE_PYTHON` before opening it from Terminal:

```bash
cd /Users/wangyiding/CellUniverse/Cell_Genesis_Studio
conda activate napari
export CELLUNIVERSE_PYTHON="$(which python)"
scripts/build_macos_app.sh
open "Dist/Cell Genesis Studio.app"
```

The Finder launched app writes its runtime log to:

```bash
tail -f /tmp/cell_genesis_studio.log
```

Manual initial cell setup now uses an embedded 3D editing workflow:

- The embedded napari 3D editor is the primary workspace. Opening a TIFF stack loads the volume into the right-side 3D panel with `viridis` as the default colormap.
- A TIFF stack can be opened either from the file picker or by dragging a local `.tif` / `.tiff` file directly onto the app window, including the embedded 3D viewer area.
- The cell kind selector starts at `Non Selected`. Choosing a cell kind creates the current editable 3D ellipsoid candidate.
- Voxel size is read from ImageJ or OME TIFF metadata when available. The voxel controls remain as an override for TIFF files that do not contain reliable physical size metadata.
- The PyQt window only keeps the dataset, category, numeric fine tuning, and save controls. The older XY/XZ/YZ projection panes are intentionally not shown in the main workflow.
- The `Open 3D Napari Adjuster` button focuses the embedded 3D viewer and synchronizes the current manual cells, red selected center, transparent ellipsoid rings, XYZ axes, and draggable axis handles. Dragging the red center updates the cell center; dragging the green X, cyan Y, or orange Z handles updates each radius.
- The ellipsoid rings have an opacity control. The XYZ axes stay bright and opaque so they remain visible over the volume.
- `W/A/S/D` moves in XY, `Q/E` moves in Z, `Shift` makes the movement step larger, `+/-` changes the selected cell size, and `Enter` finishes the selected cell.
- `Initial_frame.csv` now stores the center and three ellipsoid radii so the C++ algorithm can use the manual shape directly. `Initial_frame_manual_shapes.csv` keeps extra editing metadata for review and reload.
- A newly divided daughter pair is saved as two matched cells and also exports `Initial_frame_daughter_pairs.csv` with the pair id and center distance.

```bash
celluniverse-review review --frame 168 --run "/Volumes/T9/🦠Cell Universe/🟣Output/✅C.elegans_developing embryo 11.76G/✅168_278cells_PASS278of278_PASS_pca_center_anchor_lock_03901_20260527" --cells "/Volumes/T9/🦠Cell Universe/🟣Output/✅C.elegans_developing embryo 11.76G/✅168_278cells_PASS278of278_PASS_pca_center_anchor_lock_03901_20260527/cells.csv"
```

Important Napari note:

Dragging a 3D RGB tif into Napari is not a reliable way to review colored 3D outlines. Napari can show the thumbnail colors, but its 3D volume rendering can display the RGB stack as white or gray intensity. The reliable view uses real image layers plus one Shapes layer and one Points layer, where Napari preserves per object colors in the main 3D window.

```bash
celluniverse-review bake-overlay --frame 170 --base-tif "/Volumes/T9/🦠Cell Universe/🟣Output/✅C.elegans_developing embryo 11.76G/✅170_299cells_PASS299of299_PASS_parent_anchor_merge_shape_guard_20260527/tiff/synth/170.tif" --cells "/Users/wangyiding/CellUniverse/C++/output/F170-194_0531/✅_contains_verified_correct_frames_20260531/✅170_PASS1frames_299cells_⚠️fail171-172_recheckpoint_lateFix8_noTif_20260531/cells.csv" --output "/Volumes/T9/🦠Cell Universe/🟣Output/Visualization/_overlay_tests/170_synth_overlay_test.tif"
```
