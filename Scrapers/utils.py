import os
import time
import random
import pygame
import bs4
import urllib
import urllib.request as urllib2
import pickle
from PIL import Image
from pygame.locals import *
import scipy.ndimage
import scipy.misc
from scipy.misc import imread, imsave
from convnet import Convnet
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
batch_size = 100
dataPath = 'C:/Users/Anko/Desktop/code/nudifier/data'
nudePath = dataPath+'/nude'
dressedPath = dataPath +'/dressed'
dressedResizedPath = dataPath + '/resized/dressed'
nudeResizedPath = dataPath + '/resized/nude'
sortedPath = dataPath +'/sorted'
nudeSortedPath = sortedPath+'/nude'
nudeTrashPath = sortedPath+'/trash'
dressedSortedPath = sortedPath +'/dressed'
dressedTrashPath = sortedPath +'/trash2'
batchPath = dataPath+'/batch'
nudeBatchPath = batchPath + '/nude'
trashBatchPath = batchPath+'/trash'
modifiedPath = dataPath+'/modified'
preSortedPath = dataPath+'/pre-sorted'
preSortedNudePath = preSortedPath+'/nude'
finalPath = dataPath + '/final'
is_pygame_launched = False
allImagesUrls = []
searchedUrls = []
i_searchedUrls = 0
lastSearchedUrl = 5600
lasts = {}
imagesFound = 0
min_dim = [200,250]
def connect(url):
    req = urllib2.Request(url)
    handle = urllib2.urlopen(req)
    time.sleep(1)
    the_page = handle.read().decode()
    soup = bs4.BeautifulSoup(the_page, "html.parser")
    return the_page, soup
def getSearchedUrls():
    global allImagesUrls, searchedUrls, lastSearchedUrl
    print('azeaz')
    try:
        with open('allUrls.p', 'rb') as f:
            data = pickle.load(f)
            allImagesUrls = data['allImagesUrls']
            searchedUrls = data['searchedUrls']
    except:
        allImagesUrls = []
        searchedUrls = []
        data = {'searchedUrls':searchedUrls,'allImagesUrls':allImagesUrls}
        with open('allUrls.p','wb') as f:
            pickle.dump(data,f)
    print('poei')
    i = []
    for url in searchedUrls:
        if 'https://fr.pornhub.com/photo/' in url:
            if 1:
                i.append(int(url.split('/')[-1]))
            else: pass
    i.sort()
    try:
        lastSearchedUrl = i[-1]
    except: lastSearchedUrl = 5600
    return allImagesUrls, searchedUrls, lastSearchedUrl
def dumpUrls():
    global allImagesUrls, searchedUrls, lastSearchedUrl
    data = {'searchedUrls':searchedUrls,'allImagesUrls':allImagesUrls}
    with open('allUrls.p','wb') as f:
        pickle.dump(data,f)
        
def getImagesUrls(url):
    imgs = []
    _, soup = connect(url)
    i = soup.find_all('img')
    for img in i:
        try:
            if 'user' not in i.parent['href']:
                imgs.append(i)
        except:
            imgs.append(i)
    return imgs

def saveImageFromUrl(url, savePath, lastImage):
    global imagesFound
    if url != None and url not in allImagesUrls:
        o = 1
        while o:
            if 1:
                path = savePath+'/{}'.format(str(lastImage)+'.'+url.split('.')[-1][0:3])
                urllib2.urlretrieve(url,path)
                allImagesUrls.append(url)
                if not is_dim(path):
                    os.remove(path)
                else:
                    lastImage += 1
                    imagesFound += 1
                o = 0
            else:
                print('ERROR')
                if o > 5:
                    break
                else: o +=1
    return lastImage

def saveImagesFromUrl(url, savePath, lastImage):
    searchedUrls.append(url)
    print(url)
    try:
        _, soup = connect(url)
        print('connect')
    except: return lastImage
    imgs = soup.find_all('img')
    print(soup)
    for img in imgs:
        src = img.get('src')
        lastImage = saveImageFromUrl(src, savePath, lastImage)
    if imagesFound % 50 == 0:
        print('yeah')
        dumpUrls()
    return lastImage
def saveImagesFromSoup(soup, savePath, lastImage):
    imgs = soup.find_all('img')
    src = soup.findAll('div',{'class': 'centerImage'})[1].find('img')['src']
    print('src',src)
    lastImage = saveImageFromUrl(src, savePath, lastImage)
    if imagesFound % 50 == 0:
        print('yeah')
        dumpUrls()
    return lastImage
def findAllTags(soup):
    tags = []
    for t in soup.findAll("span", {"class": "tagLabel"}):
        tags.append(t.string)
    return tags

def findAllAlbums(url):
    if url not in searchedUrls:
        print('unknown')
        try:
            _, soup = connect(url)
        except:
            return []
        urls = []
        print('soup:',soup)
        for t in soup.findAll("div", {"class": "photoAlbumListBlock"}):
            urls.append(t.findChildren()[0]['href'])
        searchedUrls.append(url)
        return urls
    else:
        print('known')
        return []
    
def findAlbumsFromSoup(soup):
    urls = []
    for t in soup.findAll("div", {"class": "photoAlbumListBlock"}):
        urls.append(t.findChildren()[0]['href'])
    print('zzzzz', urls)
    return urls
def write(fenetre,parametres): #parametres =[texte,pos,taille,couleur]
    if len(parametres)<2 or len(parametres)>=5:
        return 0
    else:
        texte=parametres[0]
        pos=parametres[1]
        if len(parametres)==2:
            taille=20
            couleur=noir
        elif len(parametres)==3:
            taille=parametres[2]
            couleur=noir
        elif len(parametres)==4:
            taille=parametres[2]
            couleur=parametres[3]
        texte=pygame.font.Font(None,taille).render(texte,0,couleur)
        fenetre.blit(texte,pos)
        pygame.display.flip()

def launchPygame(windowSize = [800,800]):
    global is_pygame_launched
    pygame.init()
    window = pygame.display.set_mode(windowSize)
    window.fill((0,0,0))
    is_pygame_launched = 1
    return window

def getLastFromDir(path):
    i = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            try:
                i.append(int(file.split('.')[0]))
            except: pass
    i.sort()
    try:
        last = i[-1]
    except: last = 0
    return last

def moveFile(oldPath, newPath):
    global lasts
    if newPath in lasts.keys():
        nbr = lasts[newPath] + 1
    else:
        nbr = getLastFromDir(newPath) + 1 
    lasts[newPath] = nbr
    newName = str(nbr) + '.' + oldPath.split('.')[-1]
    os.rename(oldPath, newPath+'/'+newName)
    return newName

def is_dim(path):
    image = Image.open(path)
    ims = image.size
    image.close()
    if ims[0] <= min_dim[0] or ims[1] <= min_dim[1]:
        is_dim = False
    else:
        is_dim = True
    return is_dim
def redim_saved_images(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            try :
                path = subdir.replace("""\ """,'/')+'/'+file
                if not is_dim(path): os.remove(path)
            except:
                print(file)

def save_as_array(imagePath, arrayPath, size = [800,800]):
    cv = Convnet(False, False, size, form = True)
    sizeTxt = '{}x{}'.format(size[0], size[1])
    lastArray = getLastFromDir(arrayPath+'/'+sizeTxt)
    batch = []
    for subdir, dirs, files in os.walk(imagePath):
        for file in files:
            path = subdir.replace("""\ """,'/')+'/'+file
            img = imread(path).astype(np.float32)
            batch.append(img)
            if len(batch) >= batch_size:
                with open(arrayPath+'/'+sizeTxt+'/'+str(lastArray+1)+'.p', "wb") as f:
                    batch = cv.format_images(batch)
                    lastArray += 1
                    pickle.dump(batch, f)
                batch = []
    with open(arrayPath+'/'+str(lastArray+1)+'.p', "wb") as f:
        pickle.dump(batch, f)

def show_images_arrays(path):
    window = launchPygame()
    images = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            p = subdir.replace("""\ """,'/')+'/'+file
            with open(p,'rb') as f:
                batch = pickle.load(f)
            for image in batch:
                pygame.surfarray.blit_array(window,image)
                for event in pygame.event.get():pass
                pygame.display.flip()
                time.sleep(1)

def resize_images(imagePath, resizedPath, size = [800,800]):
    cv = Convnet(False, False, size, form = True)
    sizeTxt = '{}x{}'.format(size[0], size[1])
    lastResized = getLastFromDir(resizedPath+'/'+sizeTxt)
    for subdir, dirs, files in os.walk(imagePath):
        lenFiles = len(files)
        break
    nameList = list(x + lastResized for x in range(lenFiles))
    x = 0
    for subdir, dirs, files in os.walk(imagePath):
        for file in files:
            path = subdir.replace("""\ """,'/')+'/'+file
            try:
                img = imread(path).astype(np.float32)
                if img.shape[2] != 3:
                    newImg = np.zeros((img.shape[0],img.shape[1],3))
                    for x in range(len(newImg)):
                        for y in range(len(newImg[x])):
                            for z in range(len(newImg[x][y])):
                                newImg[x][y][z] = img[x][y][z]
                    img = newImg
            except Exception as e:
                print(e)
                print(file)
                continue
            img = img.reshape(img.shape[0],img.shape[1],3)
            image = cv.format_images([img])[0]
            name = nameList[random.randrange(len(nameList))]
            scipy.misc.imsave(resizedPath+'/'+sizeTxt+'/'+'{}.png'.format(x), image)
            x += 1
            nameList.remove(name)
            lastResized += 1

def invert_luminosity(path):
    last = getLastFromDir(path)
    'window = launchPygame([256,256])'
    for subdir, dirs, files in os.walk(path):
        for file in files:
            p = subdir.replace("""\ """,'/')+'/'+file
            img = imread(p).astype(np.float32)
            newImg = np.zeros(img.shape)
            for x in range(len(newImg)):
                for y in range(len(newImg[x])):
                    old = img[x][y] 
                    ttl = (np.sum(old) + 1)
                    prop = old / ttl
                    newTtl = -(ttl-765)
                    scale = [(ttl-newTtl)/3 for w in range(3)]
                    new = np.round(old - scale)
                    for w in range(3):
                        if new[w]>255: new[w] = 255
                        elif new[w]<0: new[w] = 0
                    newImg[x][y] = np.array(new)
                    """"print('old',old)
                    print('new',new)
                    print('prop',prop)
                    print('newTtl,',newTtl)
                    print(' ')"""
            """pygame.surfarray.blit_array(window,newImg)
            for event in pygame.event.get():pass
            pygame.display.flip()
            time.sleep(4)"""
            scipy.misc.imsave(path+'/'+'{}.jpg'.format(last), np.array(newImg))
            last += 1

def launchBrowser(baseUrl):
    driver = webdriver.Firefox()
    driver.get(baseUrl)
    return driver
