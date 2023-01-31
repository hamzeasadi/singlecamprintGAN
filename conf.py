import os




root = os.getcwd()
data = os.path.join(root, 'data')

paths = dict(
    root=root, data=data, iframes=os.path.join(data, 'iframes'), outpics=os.path.join(data, 'outpics'),
    videos=os.path.join(data, 'videos'), model=os.path.join(data, 'model'), model2=os.path.join(data, 'model2'),
    train=os.path.join(data, 'iframes', 'train'), val=os.path.join(data, 'iframes', 'val')
)


def rm_ds(itemlist: list):
    try:
        itemlist.remove('.DS_Store')
    except Exception as e:
        print(e)
    return itemlist

def createdir(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as e:
        print(e)
    






def main():
    pass
    


if __name__ == '__main__':
    main()