# Collecting paper-piece annotations

Use the **Multi-piece Annotator** to label photos/scans and send one collection ZIP back to Daniel. The Windows application includes its CPU model and runtime; Python, CUDA and a repository checkout are not needed.

## Start

1. Extract the entire application ZIP. Run `PaperSegmentationAnnotator.exe` from the extracted folder.
2. Choose **Open / create workspace** and select a new folder for your annotation work. Keep it outside the application folder and outside your input photo folder. Reopen this same workspace to resume.
3. Choose phone or scanner capture type. Contributor, collection, known page ID and capture session are optional. Arbitrary fragments from unrelated or unknown pages are valid training material: you never need every piece of an original page. Leave the known page ID blank when unknown. If you know fragments or repeat photos belong to one page, share its collection/page IDs with Daniel so related observations stay in the same dataset split. A session ID, when provided, must identify one real capture session rather than a generic `session1`.
4. Choose **Add photo / scan folder**. Recursive folders and multi-page PDFs are supported; PDF pages are rendered at 300 DPI. The progress line names the file being imported. If a capture has different known metadata, select it, correct the fields and choose **Save metadata**.

Sources remain at their native resolution, with EXIF orientation applied. The model uses a smaller view for suggestions; saved masks are native-size. A capture may contain zero, one or many fully visible separated pieces. Leave a visible gap between pieces and background around every piece. For phone acquisition, keep the camera parallel to the flat paper plane, at fixed height, zoom and output resolution. Scans at a consistent DPI are preferred for controlled reconstruction captures.

## Label a capture

1. Select a source. **Auto-suggest on selection** starts predictions for untouched images automatically; revisiting an annotated image reuses saved work. Turn it off to work manually, or choose **Suggest pieces** yourself. This uses the unchanged SegFormer-B1 Detail v3 model before its old one-piece filter, on CPU at 768 pixels. The progress line shows model loading, inference, separation and saving, with elapsed time. It may miss pieces or suggest too many; proposals are not approvals. Tiny omitted components are counted in the status line and can be added manually. Cancel waits for the current safe step and preserves existing work; cancelling also turns off automatic prediction until you enable it again.
2. Select a piece in the list, or double-click it, and choose **Edit selected**. Green indicates accepted pieces, amber pending proposals and blue the selected pieces in the overview.
3. In the editor, left-drag adds foreground; right-drag erases. Use the wheel to zoom, middle-drag to pan, `0` to fit, `U`/`Z` to undo, `Y` to redo, and `[` / `]` to change brush size. `S`/Enter saves the edit; Escape closes it without saving. Saving an edit returns a piece to pending review; choose **Accept** after checking it.
4. **Reject** false detections. **Add missed piece** opens an empty mask for drawing a missed fragment. You can label manually even if suggestions fail.
5. If a model bridge joined two separated pieces, erase the bridge in the editor, save, then choose **Split regions**. Select several proposals with Ctrl and choose **Merge** only when they represent parts of the same physical piece. A disconnected merged proposal still needs correction before it can be accepted.
6. When every visible real piece in this image is labeled and accepted, choose **Confirm every VISIBLE piece labeled**. This concerns only the current image, regardless of where its fragments came from. It is required for a training scene. Use **Undo** / **Redo** for saved scene changes during the current app session; saved revisions also remain on disk.

If no paper fragments are present, reject false proposals and choose **Confirm NO pieces**. This is a useful negative training example. Rejection alone does not confirm an empty image. Tick **Clean compositing background** only if the scene is also suitable as a clean background for synthetic pieces. A cluttered negative can still help training without being a clean generator background.

Use **Exclude capture** for unusable/clipped captures. Excluded captures are not negative examples. Partial work is saved but omitted from ready-to-train exports. Do not leave a real piece unlabeled in a photo marked complete: that would incorrectly teach the model it is background.

## Send the result

Choose **Export reviewed collection** and save a new ZIP outside the workspace. The summary lists reviewed scenes, pieces, confirmed empty scenes and omitted unfinished/excluded captures. Send this ZIP to Daniel using your team's normal file-sharing method. Do not send the application or its model again with each dataset.

The ZIP includes native images once per capture, a separate integer label for every accepted piece, union masks, native raster outlines in `outlines/<capture_id>.png`, metadata, checksums and a preview contact sheet. Masks are the authoritative boundaries: the owner-side `collection-assets` command uses them to extract individual real pieces for placement on synthetic backgrounds. You can keep collecting in your workspace and export another ZIP later. Daniel's importer detects repeated captures and flags conflicting revisions instead of overwriting labels.

**Publish approved pieces** is an optional export of individually accepted piece observations and their crops/masks. It never requires a complete original page. Use the reviewed collection ZIP for the training-data handoff.

## Platforms

Windows: use the complete extracted application folder. The separate macOS ZIP is a build kit and must be built on an Apple-Silicon Mac; it is not already a native `.app`. Its first build requires Python and network access. Once built, distribute the resulting native app ZIP.
