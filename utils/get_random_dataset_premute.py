import argparse
import glob
import math
import os
import random
import shutil

def split(path,train,val):
    jsons = glob.glob(os.path.join(path,"*.json"))
    if "labels.json" in jsons:
        jsons.remove("labels.json")
    random.shuffle(jsons)

    train = math.floor(len(jsons)*train/100)

    train_set = jsons[:train]
    val_set = jsons[train:]

    print(f"Whole dataset length: {len(jsons)}")
    print(f"Train dataset length: {train}")
    print(f"Test dataset length: {len(jsons) - train}")

    i = 0
    while True:
        save_dir = os.path.join(path,"..",f"dataset_split{i}")
        train_dir = os.path.join(save_dir,"train")
        val_dir = os.path.join(save_dir,"val")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            os.mkdir(train_dir)
            os.mkdir(val_dir)
            break
        i += 1
    print(f"Copying into {save_dir}...")

    dict = {train_dir:train_set,val_dir:val_set}

    for dir,set in dict.items():
        for json in set:
            base_name = os.path.basename(json)
            #Copy json
            try:
                shutil.copy(json,os.path.join(dir,base_name))
            except FileNotFoundError:
                print(f"json {base_name} is not found, but we were parsing jsons, so we didn't find a json we JUST parsed "
                      f"milliseconds ago... what happened?")
                continue
            #Copy png
            src_img_name = json[:-5] + ".png"
            src_img_base_name = os.path.basename(src_img_name)
            try:
                shutil.copy(src_img_name,os.path.join(dir,src_img_base_name))
            except:
                print(f"PNG for {base_name} is not found, skipping...")
                continue
            #Copy masks
            i = 0
            while True:
                mask_name = json[:-5] + f"_mask{i}.png"
                mask_base_name = os.path.basename(mask_name)
                try:
                    shutil.copy(mask_name,os.path.join(dir,mask_base_name))
                except FileNotFoundError:
                    if i == 0:
                        print(f"WARNING: 0 masks found for {base_name}, is this ok?")
                    break
                i += 1
    print("done.")




    #print(jsons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "A small script to split LabelME dataset directory into train and val datasets")
    parser.add_argument("dataset",help="Dataset directory with .png and .json files")
    parser.add_argument("--split",type=str,help="How dataset should be splitted in percentages (train/val)"
                                              "\nExample: 80/20 would mean 80% to train and 20% to val (default)"
                                              "\nIn case of odd split, favors val"
                        ,
                        default="80/20")

    args = parser.parse_args()

    train,val = list(map(int,args.split.split("/")))

    split(args.dataset,train,val)

