import os


# write a function to rename .jpg files inside a folder with xxxxcorrected.jpg format
def rename_names(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith('.jpg'):
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, file[:-4] + 'corrected.jpg'))
    
    print('Done renaming files in folder: ', folder_path)

def main():
    folder_path = 'results'
    rename_names(folder_path)

if __name__ == '__main__':
    main()