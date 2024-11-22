
import pickle
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from model.DynRT import *
def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret
if __name__ == "__main__":
    
    opt = {
        'data_path':'input/prepared/',
        'img_path':'dataset_image/'
    }
    id ={
        "train":load_file(opt["data_path"] + "train_id"),
        "test":load_file(opt["data_path"] + "test_id"),
        "valid":load_file(opt["data_path"] + "valid_id")
    }
    tex={
        "train":load_file(opt["data_path"] + "train_text"),
        "test":load_file(opt["data_path"] + "test_text"),
        "valid":load_file(opt["data_path"] + "valid_text")
    }
    label={
         "train":load_file(opt["data_path"] + "train_labels"),
        "test":load_file(opt["data_path"] + "test_labels"),
        "valid":load_file(opt["data_path"] + "valid_labels")

    }

    img_dir=opt["img_path"] 

    transform_train = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_valid = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform = {
        "train": transform_train,
        "valid": transform_valid,
        "test": transform_test
        }
    image_tensor = {
        "train": [],
        "valid": [],
        "test": []
    }
    save_path = 'image_tensor/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

    # Get true and false labels using the OCR function
    for mode in id.keys():
        for idx in id[mode]:
            img_path=os.path.join(
                    img_dir,
                    "{}.jpg".format(idx)
                )
            true_text,ls = process_single_image(img_path)
            if ls:
                y_pred=detect_irony(true_text,true_text)
                np.save(save_path + f"{idx}.npy", np.array(y_pred))
                print(f"Processed and saved tensor for image ID: {idx}")
            
            else:
                img = Image.open(img_path)
                img = img.convert('RGB')  # Convert grayscale to RGB if necessary
                transformed_img = transform["train"](img)  # Choose the appropriate transformation (train, valid, test)

                np.save(save_path + f"{idx}.npy", transformed_img.numpy())
                print(f"Processed and saved tensor for image ID: {idx}")


            


   
 #   true_text,true_labels, false_labels = batch_image_to_ocr_text(id["train"])
   


    # Process only the false-labeled images
    # for idx in false_labels:
    #     try:
    #         img_path = os.path.join(
    #             img_dir,
    #             f"{idx}.jpg"
    #         )
    #         img = Image.open(img_path)
    #         img = img.convert('RGB')  # Convert grayscale to RGB if necessary
    #         transformed_img = transform["train"](img)  # Choose the appropriate transformation (train, valid, test)
            
    #         # Save the tensor
    #         np.save(save_path + f"{idx}.npy", transformed_img.numpy())
    #         print(f"Processed and saved tensor for image ID: {idx}")
    #     except Exception as e:
    #         print(f"Error processing image {idx}: {e}")

    # for mode in id.keys():
    #     for idx in id[mode]:
    #         img_path=os.path.join(
    #                 img_dir,
    #                 "{}.jpg".format(idx)
    #             )
    #         img = Image.open(img_path)
    #         img = img.convert('RGB') # convert grey picture
    #         trainsform_img = transform[mode](img)
    #         image_tensor[mode].append(trainsform_img.unsqueeze(0))
    #         np.save(save_path + str(idx) + '.npy', trainsform_img.numpy())
            


    