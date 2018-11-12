from utils import *
from convnet import *
def trainConvnet(validPath, invalidPath, size = [800,800]):
    print('init')
    convnet = Convnet(True, True, size, 'model/{}x{}/'.format(size[0], size[1]))
    print('Convnet intitialized')
    validPaths  = []
    epochs = 5
    for subdir, dirs, files in os.walk(validPath):
        for file in files:
            path = subdir.replace("""\ """,'/')+'/'+file
            validPaths.append(path)
    invalidPaths = []
    print('Path found')
    for subdir, dirs, files in os.walk(invalidPath):
        for file in files:
            path = subdir.replace("""\ """,'/')+'/'+file
            invalidPaths.append(path)
    print('Invalid path found')
    for i in range(epochs):
        print('Epoch {}'.format(i))
        for x in range(min(len(validPaths),len(invalidPaths))):
            print('Batch {}'.format(x))
            try:
                with open(validPaths[x], 'rb') as f:
                    batch = pickle.load(f)
                print('Valid Batch retrived')
                valid = list([np.array([1]) for x in range(len(batch))])
                convnet.train(batch, valid, saveIters = 15)
                print('Convnet trained')
                print('Current efficiency: '+convnet.current_efficiency())
            except Exception as e: print(e)
            try:
                with open(invalidPaths[x], 'rb') as f:
                    batch = pickle.load(f)
                print('Invalid Batch retrived')
                valid = list([np.array([0]) for x in range(len(batch))])
                convnet.train(batch, valid, saveIters = 15)
                print('Convnet trained')
                print('Current efficiency: '+convnet.current_efficiency())
            except Exception as e: print(e)
        print('efficiency:',convnet.efficiency())
            
                       
            
class ManualSorter:
    def __init__(self, imagePath, sortedPath, trashPath = None, nn = Convnet, size = [800,800], train = True):
        try: nn = nn()
        except: pass
        self.window = launchPygame(size)
        self.imagePath = imagePath
        self.sortedPath = sortedPath
        self.trashPath = trashPath
        self.nn = nn
        self.train = train

    def sort(self, maxIters = 500):
        it = 0
        for subdir, dirs, files in os.walk(self.imagePath):
            for file in files[0:maxIters]:
                
                path = subdir.replace("""\ """,'/')+'/'+file
                
                if self.nn != None:
                    try:
                        imageMat = self.nn.prepare_image(path, 'path')
                        proba = self.nn.forward_only(imageMat, 'resized')
                    except:
                        print('Error',path)
                        continue
                    proba = proba[0]
                    c = [255,0,0]
                    v = np.array([[0]])
                    if proba>0:
                        c = [0,255,0]
                        v = np.array([[1]])
                        
                image = pygame.image.load(path)
                image = pygame.transform.scale(image, (800,800))
                self.window.blit(image,(0,0))
                if self.nn != None:
                    write(self.window,[str(int(proba)),[20,20],25,c])
                else: pygame.display.flip()
                continuer = 1
                while continuer:
                    for event in pygame.event.get():
                        if event.type == KEYDOWN and event.key in [K_RIGHT, K_LEFT]:
                            if event.key == K_RIGHT:
                                valid = np.array([[1]])
                                moveFile(path, self.sortedPath)
                                self.window.fill((0,255,0))
                                continuer = 0
                            else:
                                valid = np.array([[0]])
                                if self.trashPath != None:
                                    moveFile(path, self.trashPath)
                                else: os.remove(path)
                                self.window.fill((255,0,0))
                                continuer = 0
                            break
                pygame.display.flip()
                
                if self.nn != None and self.train == True:
                    cpt_loss = self.nn.train(imageMat,valid,'resized')
                    print('loss',cpt_loss)
                    if valid == np.array([[1]]):
                        imagePath = self.trashPath +'/'+ str(random.choice(os.listdir(self.trashPath)))
                        loss = self.nn.train(imagePath, np.array([[0]]),'path')
                    if valid == np.array([[0]]):
                        imagePath = self.sortedPath +'/'+ str(random.choice(os.listdir(self.sortedPath)))
                        loss = self.nn.train(imagePath, np.array([[1]]),'path')
                    print('efficiency:',self.nn.efficiency())
                it += 1
                
                
        
            
        












































        
