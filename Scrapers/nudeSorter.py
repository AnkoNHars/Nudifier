from sorter import *

def manual_nude_sorter():
    convnet = Convnet(image_size = [256,256],savePath = 'model/256x256')
    sorter = ManualSorter(nudePath, nudeSortedPath, nudeTrashPath, nn = convnet, train=False)
    sorter.sort(2000)
def manual_dressed_sorter():
    convnet = Convnet(image_size = [256,256],savePath='model/dressed')
    sorter = ManualSorter(dressedPath, dressedSortedPath, dressedTrashPath, nn = convnet, train=False)
    sorter.sort(2000)
manual_nude_sorter()
                        
