from fastbook import search_images_ddg
from fastdownload import download_url
from fastai.vision.all import *

searches = 'forest','bird'
path = Path('bird_or_not')

# That can't work in china
# for o in searches:
#     dest = (path/o)
#     dest.mkdir(exist_ok=True, parents=True)
#     download_images(dest, urls=search_images_ddg(f'{o} photo', max_images=100))
#     time.sleep(5)
#     resize_images(path/o, max_size=400, dest=path/o)

images = get_image_files(path, recurse=True)
failed = verify_images(images)
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # ImageBlock: input image, CategoryBlock: output category
    get_items=get_image_files, # get all image files in path
    splitter=RandomSplitter(valid_pct=0.2, seed=42), # split data into train and valid set, 80% for training, 20% for validation; random seed is 42 to ensure reproducibility（with the same seed, we can get the same split result every time）
    get_y=parent_label, # get the label from the parent folder name
    item_tfms=Resize(192, method = 'Squish') # resize image to 192x192, squish method will stretch the image to fit the size
).dataloaders(path, bs=32) # create a dataloader with batch size 32

learn = vision_learner(dls, resnet18, metrics=error_rate) # use resnet18 as the model architecture, and error_rate as the metric
learn.fine_tune(3) # fine tune the model for 3 epochs

is_bird, _, probs = learn.predict(PILImage.create('bird_or_not/cat.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")