import os
import shutil

path = "D:\Proyecto Final\CASIA-Iris-Interval"
dstpath = "D:\Proyecto Final\CASIA-Iris-Interval-2eyes"

dir = os.listdir(path)
lendir = len(dir)
i = 1
IFolders = 0
VFolders = 0
VNFolders = []

# Determine Folders Validation (Valid if has photo of the two eyes)
for i in range(1, lendir+1):
    numuser = '{0:0>3}'.format(i)
    newpath = path + "\\" + numuser
    LNpath = newpath + "\\L"
    RNpath = newpath + "\\R"
    LNdir = os.listdir(LNpath)
    RNdir = os.listdir(RNpath)
    if len(LNdir) == 0 or len(RNdir) == 0:
        print("Folder " + str(i) + " NOT VALID")
        IFolders += 1
    else: 
        print("Folder " + str(i) + " VALID")
        VFolders += 1
        VNFolders.append(i)


print(str(IFolders) + " Folders are NOT VALID")
print(str(VFolders) + " Folders are VALID")

# Transer VALID Folders to another Project Folder

for k in VNFolders:
    Rnumuser = '{0:0>3}'.format(k)
    Rnewpath = path + "\\" + Rnumuser
    Rdstpath = dstpath + "\\" + Rnumuser
    destination = shutil.copytree(Rnewpath, Rdstpath)

