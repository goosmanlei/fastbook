from fastai.vision.all import *

bear_types = 'grizzly','black','teddy'
path = Path('bears')

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
non_jpg = get_image_files(path).filter(lambda x: x.suffix.lower() != '.jpg')
non_jpg.map(Path.unlink)

bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

bears.new(item_tfms=Resize(128, method = ResizeMethod.Squish)).dataloaders(path).show


dls = bears.dataloaders(path)

dls.show_batch(max_n=4, nrows=1)

