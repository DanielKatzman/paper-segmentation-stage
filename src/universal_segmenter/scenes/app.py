"""Offline teammate collection application backed by the shared scene operations."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from .core import SceneStore, read_json, write_json
from .workspace import (accept_instance, add_instance, discover_into, edit_instance,
                        finish_scene, instance, merge_instances, propose,
                        reject_instance, split_instance)


class JobCancelled(Exception):
    pass


class CollectionApp:
    def __init__(self, root, project: Path | None = None, workspace: Path | None = None):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root, self.project = root, project
        self.store = self.scene = self.rgb = None
        self.segmenter = None
        self.jobs = queue.Queue()
        self.busy = False
        self.cancel_event = threading.Event()
        self._preview_key = self._preview = None
        self.undo_revisions = []
        self.redo_revisions = []
        from universal_segmenter import __version__
        root.title(f"Paper Collection — Multi-piece Annotator v{__version__}")
        root.geometry("1260x860")
        root.minsize(950,650)
        bar = ttk.Frame(root,padding=10); bar.pack(fill="x")
        ttk.Button(bar,text="Open / create workspace",command=self.choose_workspace).pack(side="left")
        ttk.Button(bar,text="Add photo / scan folder",command=self.add_folder).pack(side="left",padx=6)
        ttk.Button(bar,text="Export reviewed collection",command=self.export).pack(side="left")
        ttk.Button(bar,text="Publish approved pieces",command=self.publish).pack(side="left",padx=6)
        self.workspace_text = tk.StringVar(value="Choose a workspace outside your input folder")
        ttk.Label(root,textvariable=self.workspace_text,padding=(12,0)).pack(fill="x")
        meta = ttk.LabelFrame(root,text="Training captures — arbitrary fragments are welcome; a complete page is never required",padding=8)
        meta.pack(fill="x",padx=10,pady=6)
        self.meta = {}
        for col,(key,label,default) in enumerate([
            ("contributor","Contributor (optional)",""),("document_id","Collection (optional)",""),
            ("page_id","Known page ID (optional)",""),("capture_session_id","Capture session (optional)",""),
            ("capture_profile","Capture type","phone")]):
            ttk.Label(meta,text=label).grid(row=0,column=col,sticky="w",padx=5)
            var = tk.StringVar(value=default); self.meta[key] = var
            if key == "capture_profile":
                widget = ttk.Combobox(meta,textvariable=var,values=["phone","scanner"],state="readonly",width=12)
            else:
                widget = ttk.Entry(meta,textvariable=var,width=23)
            widget.grid(row=1,column=col,padx=5,sticky="ew")
            meta.columnconfigure(col,weight=1)
        ttk.Button(meta,text="Save metadata",command=self.save_metadata).grid(row=1,column=5,padx=5)
        body = ttk.Panedwindow(root,orient="horizontal"); body.pack(fill="both",expand=True,padx=10,pady=4)
        left = ttk.Frame(body,width=230); body.add(left,weight=0)
        ttk.Label(left,text="Source captures").pack(anchor="w")
        self.captures = tk.Listbox(left,exportselection=False,width=32)
        self.captures.pack(fill="both",expand=True)
        self.captures.bind("<<ListboxSelect>>",self.select_capture)
        ttk.Label(left,text="Piece proposals (Ctrl: select several)").pack(anchor="w",pady=(8,0))
        self.pieces = tk.Listbox(left,selectmode="extended",exportselection=False,height=12)
        self.pieces.pack(fill="x")
        self.pieces.bind("<<ListboxSelect>>",lambda _:self.draw())
        self.pieces.bind("<Double-1>",lambda _:self.edit())
        right = ttk.Frame(body); body.add(right,weight=1)
        actions = ttk.Frame(right); actions.pack(fill="x")
        for text,command in [("Suggest pieces",self.suggest),("Edit selected",self.edit),
                             ("Add missed piece",self.add_piece),("Accept",self.accept),
                             ("Reject",self.reject),("Split regions",self.split),("Merge",self.merge)]:
            ttk.Button(actions,text=text,command=command).pack(side="left",padx=2,pady=3)
        more = ttk.Frame(right); more.pack(fill="x")
        for text,command in [("Undo",self.undo),("Redo",self.redo),("Set physical ID",self.physical_id),
                             ("Exclude capture",self.exclude)]:
            ttk.Button(more,text=text,command=command).pack(side="left",padx=2,pady=3)
        self.auto_suggest = tk.BooleanVar(value=True)
        ttk.Checkbutton(more,text="Auto-suggest on selection",variable=self.auto_suggest,
                        command=self.schedule_suggestions).pack(side="left",padx=8)
        self.canvas = tk.Canvas(right,background="#202832",highlightthickness=0)
        self.canvas.pack(fill="both",expand=True)
        self.canvas.bind("<Configure>",lambda _:self.draw())
        finish = ttk.Frame(right); finish.pack(fill="x",pady=6)
        ttk.Button(finish,text="Confirm every VISIBLE piece labeled",command=self.complete).pack(side="left")
        ttk.Button(finish,text="Confirm NO pieces",command=self.empty).pack(side="left",padx=6)
        self.clean = tk.BooleanVar(value=False)
        ttk.Checkbutton(finish,text="Clean compositing background",variable=self.clean).pack(side="left")
        self.status = tk.StringVar(value="Open a workspace, add captures, and review every visible piece.")
        ttk.Label(root,textvariable=self.status,padding=10,wraplength=1200).pack(fill="x")
        job = ttk.Frame(root,padding=(10,0,10,8)); job.pack(fill="x")
        self.progress = ttk.Progressbar(job,mode="indeterminate",length=100)
        self.progress.pack(side="left")
        self.job_status = tk.StringVar(value="Idle")
        ttk.Label(job,textvariable=self.job_status).pack(side="left",padx=10)
        self.cancel_button = ttk.Button(job,text="Cancel operation",command=self.cancel_job,state="disabled")
        self.cancel_button.pack(side="right")
        self.poll_job = root.after(100,self.poll)
        root.protocol("WM_DELETE_WINDOW",self.close)
        if workspace:
            self.open_workspace(workspace)

    def error(self, error):
        from tkinter import messagebox
        messagebox.showerror("Collection needs attention",str(error),parent=self.root)

    def guarded(self, operation):
        try:
            if self.busy:
                raise ValueError("Wait for the current operation to finish")
            return operation()
        except Exception as error:
            self.error(error)

    def run_job(self, task, done, label="Working", *, cancellable=False):
        if self.busy:
            return
        self.busy = True
        self.cancel_event.clear()
        self.job_started, self.job_label = perf_counter(), label
        self.progress.start(40)
        self.cancel_button.configure(state="normal" if cancellable else "disabled")
        self.captures.configure(state="disabled")
        self.job_status.set(label)
        def work():
            try:
                self.jobs.put(("done",done,task(),None))
            except Exception as error:
                self.jobs.put(("done",done,None,error))
        threading.Thread(target=work,daemon=True).start()

    def report_progress(self, message):
        self.jobs.put(("progress",message))

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise JobCancelled()

    def cancel_job(self):
        if self.busy:
            self.cancel_event.set()
            self.job_label = "Cancelling after the current safe step; completed work is kept"
            self.cancel_button.configure(state="disabled")

    def poll(self):
        try:
            while True:
                event = self.jobs.get_nowait()
                if event[0] == "progress":
                    if not self.cancel_event.is_set():
                        self.job_label = event[1]
                    continue
                _,done,result,error = event
                elapsed = perf_counter()-self.job_started
                self.busy = False
                self.progress.stop()
                self.cancel_button.configure(state="disabled")
                self.captures.configure(state="normal")
                self.job_status.set(f"Finished in {elapsed:.1f}s")
                if isinstance(error,JobCancelled):
                    self.job_status.set("Cancelled; completed imports and existing annotations are preserved.")
                    # Do not immediately restart automatic prediction after cancellation.
                    self.auto_suggest.set(False)
                    self.refresh()
                elif error:
                    self.auto_suggest.set(False)
                    self.error(error)
                    self.job_status.set("Operation failed; existing annotations are preserved.")
                else:
                    try:
                        done(result)
                    except Exception as failure:
                        self.error(failure)
        except queue.Empty:
            pass
        if self.busy:
            self.job_status.set(f"{self.job_label} — {perf_counter()-self.job_started:.1f}s elapsed")
        self.poll_job = self.root.after(100,self.poll)

    def metadata(self):
        value = {k:v.get().strip() for k,v in self.meta.items()}
        if value["capture_profile"] not in {"phone","scanner"}:
            raise ValueError("Choose phone or scanner capture type")
        return value

    def choose_workspace(self):
        from tkinter import filedialog
        if self.busy:
            return
        path = filedialog.askdirectory(title="Choose/create an annotation workspace",parent=self.root)
        if path:
            self.guarded(lambda:self.open_workspace(Path(path)))

    def open_workspace(self,path):
        self.store = SceneStore(path)
        self.store.ensure_writable()
        self.store.root.mkdir(parents=True,exist_ok=True)
        state = self.store.root/"workspace.json"
        if state.exists():
            for k,v in read_json(state).get("metadata",{}).items():
                if k in self.meta:
                    self.meta[k].set(v)
        self.workspace_text.set(str(self.store.root))
        self.scene = self.rgb = None
        self.refresh()

    def add_folder(self):
        from tkinter import filedialog
        def start():
            if not self.store:
                raise ValueError("Choose a workspace first")
            metadata = self.metadata()
            path = filedialog.askdirectory(title="Folder containing photos / PDF scans",parent=self.root)
            if path:
                write_json(self.store.root/"workspace.json",{"metadata":metadata})
                self.status.set("Importing native-resolution captures…")
                store = self.store
                self.run_job(lambda:discover_into(store,Path(path),metadata,
                    progress=self.report_progress,check_cancelled=self.check_cancelled),
                    lambda _:self.refresh(),"Importing native-resolution captures",cancellable=True)
        self.guarded(start)

    def refresh(self, selected=None):
        if not self.store:
            return
        selected = selected or (self.scene.record["capture_id"] if self.scene else None)
        self.capture_ids = self.store.ids()
        self.capture_records = {}
        self.captures.delete(0,"end")
        for ident in self.capture_ids:
            record = self.store.record(ident)
            self.capture_records[ident] = record
            name = record["metadata"].get("source_name",ident[7:19])
            count = sum(i["state"] != "rejected" for i in record["instances"])
            self.captures.insert("end",f"{Path(name).name} | {record['state']} | {count}")
        if self.capture_ids:
            index = self.capture_ids.index(selected) if selected in self.capture_ids else 0
            self.captures.selection_set(index)
            self.select_capture()
        else:
            self.draw()

    def select_capture(self,event=None):
        if self.busy or not self.captures.curselection():
            return
        ident = self.capture_ids[self.captures.curselection()[0]]
        changed = self.scene is None or self.scene.record["capture_id"] != ident
        try:
            record = self.capture_records[ident]
            if changed or self.scene.record["revision"] != record["revision"]:
                self.scene = self.store.load(ident)
            if changed:
                self.rgb = self.store.rgb(self.scene)
            if changed:
                self.undo_revisions.clear(); self.redo_revisions.clear()
            for k,v in self.meta.items():
                v.set(self.scene.record["metadata"].get(k,v.get()))
            self.clean.set(self.scene.record.get("clean_background",False))
            self.pieces.delete(0,"end")
            self.piece_ids = []
            for i,item in enumerate(self.scene.record["instances"]):
                self.piece_ids.append(item["instance_id"])
                self.pieces.insert("end",f"{i+1}: {item['state']}  {item['instance_id'][-6:]}")
            proposal = self.scene.record.get("proposal") or {}
            hint = ("Waiting for automatic suggestions." if self.auto_suggest.get() else
                    "Idle — choose Suggest pieces or Add missed piece.") if not proposal and not self.scene.active and self.scene.record["state"] == "in_progress" else (
                    "Review proposals, add missed pieces, then confirm every VISIBLE piece in this image.")
            timings = proposal.get("timings_ms",{})
            speed = f" Model inference: {timings['inference']/1000:.1f}s." if "inference" in timings else ""
            self.status.set(f"{self.scene.record['state']}: {len(self.scene.active)} active pieces. "
                f"{len(proposal.get('excluded_specks',[]))} tiny components omitted from suggestions. "
                f"{hint}{speed}")
            self.draw()
            self.schedule_suggestions()
        except Exception as error:
            self.error(error)

    def draw(self):
        from PIL import ImageTk
        self.canvas.delete("all")
        if self.scene is None:
            return
        width, height = max(1,self.canvas.winfo_width()-20),max(1,self.canvas.winfo_height()-20)
        scale = min(width/self.rgb.shape[1],height/self.rgb.shape[0],1.)
        size = (max(1,round(self.rgb.shape[1]*scale)),max(1,round(self.rgb.shape[0]*scale)))
        key = (self.scene.record["capture_id"],self.scene.record["revision"],size)
        if key != self._preview_key:
            image = Image.fromarray(self.rgb).resize(size,Image.Resampling.BILINEAR)
            labels = np.asarray(Image.fromarray(self.scene.labels).resize(size,Image.Resampling.NEAREST))
            self._preview = (np.asarray(image),labels)
            self._preview_key = key
        preview, labels = self._preview
        shown = preview.copy()
        selected = set(self.selected())
        for item in self.scene.active:
            mask = labels==item["label"]
            color = np.array((50,240,100) if item["state"]=="accepted" else (255,180,50))
            if item["instance_id"] in selected:
                color = np.array((70,180,255))
            shown[mask] = (.65*shown[mask]+.35*color).astype(np.uint8)
        image = Image.fromarray(shown)
        painter = ImageDraw.Draw(image)
        for i,item in enumerate(self.scene.record["instances"]):
            if item["state"]=="rejected":
                continue
            x,y,w,h = item["bbox_xywh"]
            painter.rectangle((x*scale,y*scale,(x+w)*scale,(y+h)*scale),outline="#00aaff",width=2)
            painter.text((x*scale+3,y*scale+3),str(i+1),fill="red",stroke_width=1,stroke_fill="white")
        self.photo = ImageTk.PhotoImage(image,master=self.root)
        self.canvas.create_image(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2,
                                 image=self.photo,anchor="center")

    def selected(self):
        return [self.piece_ids[i] for i in self.pieces.curselection()] if hasattr(self,"piece_ids") else []

    def require_scene(self):
        if self.scene is None:
            raise ValueError("Choose a source capture first")

    def commit(self,operation):
        def execute():
            self.require_scene()
            updated = self.scene.copy()
            result = operation(updated)
            self.undo_revisions.append(self.scene.record["revision"])
            self.redo_revisions.clear()
            self.store.save(updated)
            self.scene = updated
            self.refresh()
            return result
        return self.guarded(execute)

    def save_metadata(self):
        self.commit(lambda s:s.record["metadata"].update(self.metadata()))

    def schedule_suggestions(self):
        if not self.busy and self.scene is not None and self.auto_suggest.get():
            capture = self.scene.record["capture_id"]
            def start_if_current():
                if (not self.busy and self.auto_suggest.get() and self.scene is not None
                        and self.scene.record["capture_id"] == capture
                        and not self.scene.record["instances"] and not self.scene.record.get("proposal")
                        and self.scene.record["state"] == "in_progress"):
                    self.suggest()
            self.root.after_idle(start_if_current)

    def suggest(self):
        def start():
            self.require_scene()
            if self.scene.record["instances"] or self.scene.eligible:
                raise ValueError("Suggestions are available for untouched captures. Existing work is preserved.")
            scene,rgb,store = self.scene.copy(),self.rgb,self.store
            def task():
                self.check_cancelled()
                if self.segmenter is None:
                    self.report_progress("Loading SegFormer-B1 Detail v3 (CPU)")
                    from universal_segmenter.inference.release import Segmenter
                    self.segmenter = Segmenter(project=self.project,cpu_threads=4)
                self.check_cancelled()
                self.report_progress("Segmenting image with the v3 model (CPU)")
                probability,_,timings = self.segmenter.probability(rgb)
                self.check_cancelled()
                self.report_progress("Separating piece proposals")
                propose(scene,probability,self.segmenter.release.threshold,
                        self.segmenter.release.cache_namespace)
                self.check_cancelled()
                self.report_progress("Saving native-resolution proposals")
                scene.record["proposal"]["timings_ms"] = timings
                store.save(scene)
                return scene
            def done(result):
                self.scene = result
                self.refresh()
            self.run_job(task,done,"Starting SegFormer-B1 Detail v3 suggestions",cancellable=True)
        self.guarded(start)

    def edit(self,adding=False):
        def start():
            from universal_segmenter.annotation.editor import MaskEditor
            self.require_scene()
            identities = self.selected()
            if not adding and len(identities)!=1:
                raise ValueError("Select exactly one piece to edit")
            ident = identities[0] if not adding else None
            mask = np.zeros(self.scene.labels.shape,bool) if adding else self.scene.labels==instance(self.scene,ident)["label"]
            editor = MaskEditor(self.rgb,mask,mask,"Add missed piece" if adding else "Edit one paper piece",
                parent=self.root,auto_regularize=False,keep_largest_on_save=False,
                dataset_split="unassigned",dataset_split_choices=[("unassigned","Assigned centrally")])
            action,corrected = editor.run()
            if action=="save":
                self.commit(lambda s:add_instance(s,corrected) if adding else edit_instance(s,ident,corrected))
        self.guarded(start)

    def add_piece(self): self.edit(adding=True)
    def accept(self):
        def operation(scene):
            if not self.selected():
                raise ValueError("Select one or more pieces to accept")
            for ident in self.selected():
                accept_instance(scene,ident)
        self.commit(operation)
    def reject(self):
        def operation(scene):
            if not self.selected():
                raise ValueError("Select one or more pieces to reject")
            for ident in self.selected():
                reject_instance(scene,ident)
        self.commit(operation)
    def split(self):
        def operation(scene):
            if len(self.selected())!=1:
                raise ValueError("Select one piece to split")
            split_instance(scene,self.selected()[0])
        self.commit(operation)
    def merge(self):
        self.commit(lambda s:merge_instances(s,self.selected()))
    def complete(self):
        from tkinter import messagebox
        if messagebox.askyesno("Review this image","Have you labeled and reviewed EVERY VISIBLE paper piece in this image?\n"
                "The fragments can be unrelated; a complete original page is not required.",parent=self.root):
            self.commit(lambda s:finish_scene(s))
    def empty(self):
        from tkinter import messagebox
        if messagebox.askyesno("Confirm empty capture","Does the source contain NO real paper pieces? Rejected proposals alone do not prove this.",parent=self.root):
            self.commit(lambda s:finish_scene(s,empty=True,clean_background=self.clean.get()))
    def exclude(self):
        self.commit(lambda s:s.record.update(state="excluded",clean_background=False))

    def physical_id(self):
        from tkinter import simpledialog
        if len(self.selected())==1:
            value = simpledialog.askstring("Physical identity","Optional physical-piece ID shared across repeat photographs:",parent=self.root)
            if value is not None:
                self.commit(lambda s:instance(s,self.selected()[0]).update(physical_piece_id=value.strip()))

    def undo(self):
        def execute():
            if self.scene and self.undo_revisions:
                revision = self.undo_revisions.pop()
                self.redo_revisions.append(self.scene.record["revision"])
                self.scene = self.store.restore(self.scene.record["capture_id"],revision)
                self.refresh()
        self.guarded(execute)
    def redo(self):
        def execute():
            if self.scene and self.redo_revisions:
                revision = self.redo_revisions.pop()
                self.undo_revisions.append(self.scene.record["revision"])
                self.scene = self.store.restore(self.scene.record["capture_id"],revision)
                self.refresh()
        self.guarded(execute)

    def export(self):
        from tkinter import filedialog,messagebox
        from .exchange import export_collection
        def start():
            if not self.store:
                raise ValueError("Open a workspace first")
            path = filedialog.asksaveasfilename(title="Export reviewed collection",defaultextension=".zip",parent=self.root)
            if path:
                store = self.store
                self.run_job(lambda:export_collection(store,Path(path),progress=self.report_progress,
                    check_cancelled=self.check_cancelled),lambda r:messagebox.showinfo(
                    "Collection exported",f"{r['scene_count']} reviewed scenes, {r['instance_count']} pieces, "
                    f"{r['empty_count']} empty scenes.\n{r['unfinished_omitted']} unfinished/excluded scenes omitted.\n{path}",parent=self.root),
                    "Exporting reviewed training collection",cancellable=True)
        self.guarded(start)

    def publish(self):
        from tkinter import filedialog,messagebox
        from .workspace import publish_pieces
        def start():
            if not self.store:
                raise ValueError("Open a workspace first")
            parent = filedialog.askdirectory(title="Parent folder for a NEW approved-pieces publication",parent=self.root)
            if parent:
                from tkinter import simpledialog
                name = simpledialog.askstring("New publication","New folder name:",initialvalue="stage1_approved",parent=self.root)
                if name and Path(name).name==name:
                    store = self.store
                    self.run_job(lambda:publish_pieces(store,Path(parent)/name),lambda r:messagebox.showinfo(
                        "Approved pieces published",f"{r['approved_piece_count']} approved pieces. "
                        "Each accepted observation is exported independently; no complete page is required.",parent=self.root),
                        "Publishing accepted piece observations")
        self.guarded(start)

    def close(self):
        from tkinter import messagebox
        if self.busy:
            messagebox.showinfo("Operation running","Wait for the current import, prediction or export to finish before closing.",parent=self.root)
        else:
            self.root.after_cancel(self.poll_job)
            self.root.destroy()


def launch(project: Path | None = None, workspace: Path | None = None):
    import tkinter as tk
    root = tk.Tk()
    CollectionApp(root,project=project,workspace=workspace)
    root.mainloop()
