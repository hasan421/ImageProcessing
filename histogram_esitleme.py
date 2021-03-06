
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
def histogram(img,L):
    histogram_sonucu,bind=np.histogram(img,bins=L,range=(0,L))
    return histogram_sonucu

def normalize_histogram(img,L):
    histogram_img=histogram(img,L)
    print(img.size)
    return  histogram_img/img.size
    


def kümülatif_hist(kümülatif):
    return np.cumsum(kümülatif)
    
def yeni_deger(img,L,histogram):
    kümülatif_histogram=kümülatif_hist(histogram)
    donüsüm_fonksiyonu=(L-1)*kümülatif_histogram
    shape=img.shape
    ravel=img.ravel()
    hist_ravel=np.zeros_like(ravel)
    for i, pixsel in enumerate(ravel):
        hist_ravel[i]=donüsüm_fonksiyonu[pixsel]
    return hist_ravel.reshape(shape).astype(np.uint8)

def main():
    img=cv.imread('/home/hasan/Desktop/mavi/4/57-0.jpg',0)
    L=2**8
    histogram= normalize_histogram(img,L)
    normalize_img=yeni_deger(img,L,histogram)
    hazir_fonksiyon=cv.equalizeHist(img)
   
    plt.hist(normalize_img.ravel(),256,[0,256]); plt.show()
    #plt.hist(hazir_fonksiyon.ravel(),256,[0,256]); plt.show()


   # cv.imshow('histogram',hazir_fonksiyon)
    
    cv.waitKey(0)

main()


