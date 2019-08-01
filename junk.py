import os

files = os.listdir('/home/sadams/sites/MezSnake/Images')
# print(files)

files2 = [i for i in files if 'png' in i and 'thermal' not in i and 'intensity' not in i]
print('\n'.join(files2))

print("\n\n")
newFiles = []
for file in files:
    if 'png' in file and 'thermal' not in file and 'intensity' not in file and len(file) > 15:
        newFiles.append(file)
print('\n'.join(newFiles))

mylist = ['red', 'green', 'blue', 'yellow']
for color in mylist:
    print(color)