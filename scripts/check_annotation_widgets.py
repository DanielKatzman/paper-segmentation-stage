"""Own-application Tk widget control, separate from desktop screenshot inspection."""
import sys
import tempfile
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def main():
    import argparse
    import tkinter as tk
    import numpy as np
    from universal_segmenter.annotation.editor import MaskEditor
    from universal_segmenter.scenes.app import CollectionApp
    from universal_segmenter.scenes.core import SceneStore,write_json
    from universal_segmenter.scenes.workspace import add_instance,finish_scene
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path,nargs="?",default=Path("exports/widget_control.json"))
    parser.add_argument("--photo",type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="paper-widget-control-") as folder:
        store=SceneStore(Path(folder)/"workspace")
        rgb=np.full((160,240,3),80,np.uint8)
        scene=store.add(rgb,{"contributor":"widget-fixture","document_id":"widget",
            "page_id":"page","capture_session_id":"session","capture_profile":"scanner"})
        for x in [20,140]:
            mask=np.zeros((160,240),bool);mask[20:130,x:x+50]=True
            add_instance(scene,mask,accepted=True,origin="widget_fixture")
        finish_scene(scene);store.save(scene)
        root=tk.Tk()
        try:
            app=CollectionApp(root,workspace=store.root)
            def fail(error):
                raise RuntimeError(str(error))
            app.error=fail
            root.update()
            assert app.captures.size()==1 and app.pieces.size()==2
            app.pieces.selection_set(0)
            selected=app.selected()[0]
            other=app.scene.labels==app.scene.active[1]["label"]
            app.reject()
            assert len(app.scene.active)==1
            app.undo()
            assert len(app.scene.active)==2
            app.redo()
            assert len(app.scene.active)==1
            app.undo()
            assert np.array_equal(app.scene.labels==app.scene.active[1]["label"],other)
            mask=app.scene.labels==app.scene.active[0]["label"]
            editor=MaskEditor(app.rgb,mask,mask,"Widget editor control",parent=root,
                auto_regularize=False,keep_largest_on_save=False,
                dataset_split="unassigned",dataset_split_choices=[("unassigned","Assigned centrally")])
            root.after(200,lambda:editor._finish("save"))
            action,corrected=editor.run()
            assert action=="save" and np.array_equal(corrected,mask)
            app.draw();root.update_idletasks()
            # Untouched captures start automatically. Cancelling must leave the
            # original revision untouched, and a successful retry is saved once.
            untouched=store.add(np.full((160,240,3),81,np.uint8),{"capture_profile":"phone"})
            entered,release=threading.Event(),threading.Event()
            calls=[]
            def probability(rgb):
                calls.append(1);entered.set()
                assert release.wait(5),"Fixture inference was never released"
                return mask.astype(np.float32)*.9,{}, {"inference":1.0}
            app.segmenter=SimpleNamespace(release=SimpleNamespace(threshold=.3,cache_namespace="fixture"),
                                           probability=probability)
            def wait_for(condition):
                deadline=monotonic()+10
                while not condition():
                    root.update()
                    assert monotonic()<deadline,"Own-application job timed out"
                    sleep(.01)
                root.update()
            app.refresh(untouched.record["capture_id"])
            wait_for(lambda:app.busy and entered.is_set())
            assert str(app.captures.cget("state"))=="disabled"
            app.cancel_job();release.set()
            wait_for(lambda:not app.busy)
            assert not app.auto_suggest.get()
            assert store.load(untouched.record["capture_id"]).record["revision"]==untouched.record["revision"]
            app.auto_suggest.set(True);app.schedule_suggestions()
            wait_for(lambda:len(calls)==2 and not app.busy)
            assert len(app.scene.active)==1 and app.scene.record["proposal"]
            assert store.load(untouched.record["capture_id"]).record["proposal"]
            root.update()
            # Refreshing an unchanged selection must not decode any source masks.
            original_load=app.store.load
            def unexpected_load(*args,**kwargs):
                raise AssertionError("Refreshing a cached capture decoded native masks")
            app.store.load=unexpected_load
            app.refresh();root.update()
            app.store.load=original_load
            assert len(calls)==2
            report={"status":"passed","scope":"Tk widget construction and own-app callbacks; not a desktop screenshot audit",
                    "scene_load":True,"overview_render":True,"independent_review":True,
                    "undo_redo":True,"modal_native_editor":True,"automatic_prediction":True,
                    "safe_prediction_cancel":True,"saved_prediction_reuse":True,"metadata_only_refresh":True}
        finally:
            for pending in root.tk.call("after","info"):
                root.after_cancel(pending)
            root.destroy()
    if args.photo:
        from PIL import Image,ImageOps
        with tempfile.TemporaryDirectory(prefix="paper-native-photo-control-") as folder:
            native=SceneStore(Path(folder)/"workspace")
            with Image.open(args.photo) as image:
                photo=np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
            source=native.add(photo,{"capture_profile":"phone"})
            root=tk.Tk()
            try:
                started=monotonic()
                app=CollectionApp(root,workspace=native.root)
                app.error=fail
                while not app.scene.record.get("proposal") or app.busy:
                    root.update()
                    assert monotonic()-started<45,"Native photo prediction timed out"
                    sleep(.01)
                elapsed=monotonic()-started
                started=monotonic()
                for _ in range(10):
                    app.refresh();app.draw();root.update_idletasks()
                report["native_photo"]={"shape":list(photo.shape),"pieces_proposed":len(app.scene.active),
                    "load_predict_save_preview_seconds":elapsed,
                    "ten_cached_refresh_and_draw_seconds":monotonic()-started,
                    "proposal_saved":bool(native.load(source.record["capture_id"]).record["proposal"])}
            finally:
                for pending in root.tk.call("after","info"):
                    root.after_cancel(pending)
                root.destroy()
    write_json(args.output,report)
    print(report)


if __name__=="__main__":
    main()
