import os

mypath = 'MIT_Dataset/Videos/'
onlyfiles = [('MIT_Dataset/Videos/'+ f, f) for f in os.listdir(mypath)]

processed = 'processed/'
processedfiles = [f.split('.')[0] for f in os.listdir(processed)]
print(processedfiles)

for file, f in onlyfiles:
    if not f.split('.')[0] in processedfiles:
        os.system('openFace/OpenFace/build/bin/FeatureExtraction -f {file} -aus'.format(file=file))