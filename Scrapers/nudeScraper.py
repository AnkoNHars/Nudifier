from utils import *


def retrive_pornhub():
    
    getSearchedUrls()
    print(allImagesUrls[0:100], searchedUrls[0:100])
    lastImage = getLastFromDir(nudePath)
    for x in range(50000):
        x += lastSearchedUrl
        url = 'https://fr.pornhub.com/photo/'+str(x+30000)
        lastImage = saveImagesFromUrl(url, nudePath, lastImage)
            
def retrive_albums(customSearch, exitTags = []):
    global allImagesUrl, searchedUrls, lastSearchedUrl
    print('kkk')
    allImagesUrls, searchedUrls, lastSearchedUrl = getSearchedUrls()
    print('ooo')
    print(allImagesUrls[0:5], searchedUrls[0:5])
    print(customSearch)
    albums = findAllAlbums(customSearch)
    print(albums)
    lastImage = getLastFromDir(nudePath)
    for album in albums:
        url = 'https://fr.pornhub.com'+album
        print(url)
        _,soup = connect(url)
        tags = findAllTags(soup)
        if len(list(set(tags).intersection(exitTags)))==0:
            albums2 = findAlbumsFromSoup(soup)
            for album2 in albums2:
                url = 'https://fr.pornhub.com'+album2
                lastImage = saveImagesFromUrl(url,nudePath,lastImage)


#retrive_albums('https://fr.pornhub.com/albums/female?search=naked+model&page=15',['men','gay','dick','cock','big cock','muscle','muscle man','man','male','penis'])

        

            
                                     
