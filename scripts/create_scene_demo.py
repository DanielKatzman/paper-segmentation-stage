"""Create self-owned example sources, scene truth and a portable collection."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def main():
    import numpy as np
    import pymupdf
    from PIL import Image
    from universal_segmenter.scenes.core import SceneStore,file_hash,write_json
    from universal_segmenter.scenes.generator import GeneratorConfig,generate
    from universal_segmenter.scenes.exchange import export_collection
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    root=args.output.resolve()
    if root.exists():
        raise ValueError("Demo output must be a new folder")
    root.mkdir(parents=True)
    source=root/"sources"; source.mkdir()
    bg=source/"background.png"
    rng=np.random.default_rng(42)
    pixels=np.clip(np.array([65,85,100])+rng.normal(0,3,(800,800,3)),0,255).astype(np.uint8)
    Image.fromarray(pixels).save(bg)
    pdf_path=source/"example_page.pdf"
    with pymupdf.open() as pdf:
        page=pdf.new_page(width=400,height=560)
        page.insert_text((35,45),"PAPER RECONSTRUCTION",fontsize=19)
        for line in range(26):
            page.insert_text((35,80+line*15),f"Example printed content. Line {line+1}. Shapes and edges matter.",fontsize=9)
        page.draw_rect((40,460,190,520),color=(.1,.2,.4),width=2)
        pdf.save(pdf_path)
    inventory=source/"inventory.json"
    write_json(inventory,{"schema_version":"paper-scene-source-inventory-1.0","assets":[
        {"kind":"background","path":str(bg),"sha256":file_hash(bg),"capture_profile":"scanner","split":"train"},
        {"kind":"background","path":str(bg),"sha256":file_hash(bg),"capture_profile":"phone","split":"train"},
        {"kind":"pdf","path":str(pdf_path),"sha256":file_hash(pdf_path),"document_id":"demo-owned-page","split":"train"}]})
    config=GeneratorConfig(scene_count=8,master_size=768,training_size=768,
        source_weights={"procedural":1},count_bins=[[0,0,.2],[1,1,.1],[4,4,.4],[8,8,.3]])
    generate(inventory,root/"workspace",config)
    store=SceneStore(root/"workspace")
    raw=root/"raw_captures"; raw.mkdir()
    for number,ident in enumerate(store.ids(),1):
        scene=store.load(ident)
        Image.fromarray(store.rgb(scene)).save(raw/f"capture_{number:02d}.png")
    export_collection(store,root/"example_collection.zip")
    print(root)


if __name__=="__main__":
    main()
