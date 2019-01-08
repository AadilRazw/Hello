# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:45:43 2018

@author: ljr
"""

import numpy as np
import cv2 as cv
#import video
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import pymssql
import pandas as pd
import datetime
from latlon import latlon_read   ##路径：D:\数据读取
from radar import readradar
from read_pup import * 

def calc_scale_and_offset(min_v, max_v, n=16):
    # stretch/compress data to the avaiable packed range
    if max_v - min_v == 0:
        scale_factor = 1.0
        add_offset = min_v
    else:
        scale_factor = (max_v - min_v) / (2**n -1)
        # translate the range to be symmetric about zero
        add_offset = min_v + 2 ** (n -1) * scale_factor
    return scale_factor, add_offset

class opt_flow():
    def __init__(self, pic1_path, pic2_path, res=0.01):
        self.pic1_path = pic1_path
        self.pic2_path = pic2_path
        a =  ReadPup(self.pic1_path)        
        a.read_data()
        convert_coord(a)
        a1 = a.JWColor
        self.a1 = a1
        b =  ReadPup(self.pic2_path)        
        b.read_data()
        convert_coord(b)
        b1 = b.JWColor
        self.b1 = b1
        print('='*30)
        print(np.mean(a1))
        print(np.mean(b1))
        a1 = np.log(a1)*10
        a1[a1<10]=0
        a2 = ((a1-np.min(a1))/np.max([np.max(a1),np.max(b1)]))*255
        a3 = a2.astype(np.uint8).reshape(a1.shape[0],a1.shape[1],1)
        self.a4 = np.concatenate((a3,a3,a3),axis=2)
       
        b1 = np.log(b1)*10
        b1[b1<10]=0
        b2 = ((b1-np.min(b1))/np.max([np.max(a1),np.max(b1)]))*255
        b3 = b2.astype(np.uint8).reshape(b1.shape[0],b1.shape[1],1)
        self.b4 = np.concatenate((b3,b3,b3),axis=2)
        
#        self.lon = np.arange(np.min(a.J), np.max(a.J) + res/2, res)
#        self.lat = np.arange(np.min(a.W), np.max(a.W) + res/2, res) 
        
        self.lon = a.J
        self.lat = a.W
        self.res = res

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis
        
    def draw_hsv(self, svname):
        h, w = self.flow.shape[:2]
        fx, fy = self.flow[:,:,0], self.flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        self.v = v
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        self.sv2nc(v, svname+'.nc')
        return bgr
        
    def warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv.remap(img, flow, None, cv.INTER_LINEAR)
        return res

    def pic(self):
        prevgray = cv.cvtColor(self.a4, cv.COLOR_BGR2GRAY)   
        gray = cv.cvtColor(self.b4, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 50, 3, 5, 1.2, 0)  
        self.flow = flow
        '''
        (输入前一帧图像; 输入后一帧图像; 输出的光流; 金字塔上下两层之间的尺度关系; 金字塔层数; 
        均值窗口大小，越大越能denoise并且能够检测快速移动目标，但会引起模糊运动区域; 
         迭代次数; 像素领域大小，一般为5，7等; 高斯标注差，一般为1-1.5; 
         计算方法。主要包括OPTFLOW_USE_INITIAL_FLOW和OPTFLOW_FARNEBACK_GAUSSIAN)
        '''
#        %matplotlib tk
        Q = plt.quiver(flow[::15,::15,0], flow[::15,::15,1])
        qk = plt.quiverkey(Q, 0.5, 0.98, 2, r'$2 \frac{m}{s}$', labelpos='W',
                           fontproperties={'weight': 'bold'})
        plt.gca().invert_yaxis()
        l, r, b, t = plt.axis()
        dx, dy = r - l, t - b
        plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])
        plt.axis('off')
        plt.title('window width=50')
        plt.show()
        #plt.savefig('C:/Users/Administrator/Desktop/光流法/opt/外推/flow.png')
        
    def density(self, df, svname):
#        for i,lat1 in enumerate(df.lat):
#            if not isinstance(lat1, float):
#                print(lat1, type(lat1))
        df[['lat', 'lon', 'inte']] = df[['lat', 'lon', 'inte']].astype(float)
        df = df[(df['lat'] >= np.min(self.lat)) &
                (df['lat'] <= np.max(self.lat)) &
                (df['lon'] >= np.min(self.lon)) &
                (df['lon'] <= np.max(self.lon))]
        df['x'] = np.round((df.lat - np.min(self.lat))/self.res).astype('int')
        df['y'] = np.round((df.lon - np.min(self.lon))/self.res).astype('int')
        self.lightning = df
        self.dfn = df.groupby(['x','y'])['x'].count()
        if not self.dfn.empty:
            print(df.iloc[:,0].size)
#            print(self.dfn)
#            print(self.lon)
#            print(self.lat)
            indx = self.dfn.index.get_level_values('x')
            indy = self.dfn.index.get_level_values('y')
            df['inten'] = np.abs(df.inte)
            dfn1 = df.groupby(['x','y'])['inte'].mean()
            self.data = np.zeros([len(self.lat), len(self.lon)])
            data1 = self.data.copy()
            data1[indx, indy] = dfn1.values
            nn = 10
            data1 = cv.blur(data1,(nn,nn))*nn*nn
            self.data[indx, indy] = self.dfn.values
            self.data = cv.blur(self.data,(nn,nn))*nn*nn
            data1[self.data>0] = data1[self.data>0]/self.data[self.data>0]
            data1[self.data<=0] = 0
            self.data1 = data1
#            print(np.max(self.data))
#            print(svname)
#            print(self.res)
            self.data1 = cv.blur(data1,(5,5))
            self.data = cv.blur(self.data,(5,5))
            self.sv2nc(self.data, svname+'_Density_lightning.nc')
            self.sv2nc(self.data1, svname+'_Intensity_lightning.nc')
            indx = np.where(self.data == 0)
            self.flow_v = self.v.copy()
            self.flow_v[indx] = 0
            self.flow_v = cv.blur(self.flow_v,(5,5))
            self.sv2nc(self.flow_v, svname+'_flow.nc')
        
    def sv2nc(self, var, svname):
        ds = xr.Dataset()
        ds.coords['lon'] = ('lon', self.lon)
        ds['lon'].attrs['units'] = "degrees_east"
        ds['lon'].attrs['long_name'] = "Longitude"
        ds.coords['lat'] = ('lat', self.lat)
        ds['lat'].attrs['units'] = "degrees_north"
        ds['lat'].attrs['long_name'] = "Latitude"
        scale_factor, add_offset = calc_scale_and_offset(np.min(var),
                        np.max(var))
        var = np.short((var - add_offset) / scale_factor)
        ds['var'] = (('lat', 'lon'), var)
        ds['var'].attrs['add_offset'] = add_offset
        ds['var'].attrs['scale_factor'] = scale_factor
        ds.to_netcdf(svname, format='NETCDF3_CLASSIC')    
        
    def read_light(self, light_path):       
        url = os.path.join(light_path, self.pic1_path[-25:-21], 
                           self.pic1_path[-25:-21]+'_'+self.pic1_path[-21:-19]+'_'+self.pic1_path[-19:-17]+'.txt')
        if os.path.exists(url):
            df = pd.read_csv(url, sep='\s+', header=None, encoding='gbk')         
            coln = ['lat','lon','inte','steep','err','loc','prov','city','county']
            spls = ['=', ':']
            dfn = pd.DataFrame(index=pd.to_datetime(df.iloc[:,1] + ' ' + df.iloc[:,2], format='%Y-%m-%d %H:%M:%S.%f'))
            for i in range(len(coln)):
                if i < 5:
                    dfn[coln[i]] = list(map(float, map(lambda x:x.split(spls[0])[-1],df.iloc[:,3+i])))
                else:
                    dfn[coln[i]] = list(map(lambda x:x.split(spls[1])[-1],df.iloc[:,3+i]))      
            st_time = datetime.datetime(int(self.pic1_path[-25:-21]), int(self.pic1_path[-21:-19]), int(self.pic1_path[-19:-17]), 
                                        int(self.pic1_path[-16:-14]), int(self.pic1_path[-14:-12]), int(self.pic1_path[-12:-10]))
            en_time = datetime.datetime(int(self.pic2_path[-25:-21]), int(self.pic2_path[-21:-19]), int(self.pic2_path[-19:-17]), 
                                        int(self.pic2_path[-16:-14]), int(self.pic2_path[-14:-12]), int(self.pic2_path[-12:-10])) 
            dfn = dfn[(dfn.index>=st_time) & (dfn.index<=en_time)]
        else:
            dfn = pd.DataFrame()
        return dfn
          
#a = r'D:\光流法\数据\R\文件7\20170430.065543.04.19.971'
#b = r'D:\光流法\数据\R\文件7\20170622.070113.04.19.971'     
#st_time = datetime.datetime(int(a[-25:-21]), int(a[-21:-19]), int(a[-19:-17]), int(a[-16:-14]), int(a[-14:-12]), int(a[-12:-10]))            
#en_time = datetime.datetime(int(b[-25:-21]), int(b[-21:-19]), int(b[-19:-17]), int(b[-16:-14]), int(b[-14:-12]), int(b[-12:-10]))
#sta_time = st_time.strftime("%Y-%m-%d %H:%M:%S")
#end_time = en_time.strftime("%Y-%m-%d %H:%M:%S")                
#table='Flash'
#conn = pymssql.connect(
#        host='192.168.1.133',
#        user='lls', 
#        password='Aa123456',
#        database='lightning')
#cur = conn.cursor()
#cur.execute(
#        "SELECT * FROM [lightning].[dbo].[%s]  \
#        WHERE [时间]>='%s' and [时间]<='%s'" %(table, sta_time, end_time))
#flash = cur.fetchall()
#columns = [i[0] for i in cur.description]
#flash = pd.DataFrame(flash,columns=columns)
#flash = flash[(flash['定位站数']>=3) & ((flash['电流']>2) | (flash['电流']<-2)) & (flash['回击'] > 0)]    
#conn.close() 
#       
#%%
if __name__ == '__main__': 
    datapath = r'X:\驻马店雷达产品资料\雷电16.08.25\0824-26雷电\20160825\R\19'
    savepath = r'D:\光流法\光流闪电'
    path_list = os.listdir(datapath)
    path_list = [i for i in path_list if i[-9:-7]=='02']
    for i in range(len(path_list)-1):
        pic1_path = os.path.join(datapath, path_list[i])
        pic2_path = os.path.join(datapath, path_list[i+1])
        opt = opt_flow(pic1_path, pic2_path)
        opt.pic()
        df = opt.read_light(r'D:\light') 
        total_savepath = os.path.join(savepath, path_list[i][-25:-10])
        if np.mean(opt.a1) >= 1:
            if not df.empty:
                if df.iloc[:,0].size > 10:
                    opt.draw_hsv(total_savepath)
                    opt.density(df, total_savepath)
#                    v = opt.flow_v.flatten()
#                    data = opt.data.flatten()
#                    cor = np.corrcoef(v, data)
#                    print(cor)

#from latlon import latlon_read
#a = latlon_read(r'Z:\回波强度\MOSAICHREF000.20181111.011000.latlon', [])
#a.data 

#    rootdir = 'D:\光流法\数据'
#    savepath = 'D:\光流法\光流-闪电'
#    list = os.listdir(rootdir)
#    for i,j,k in os.walk(rootdir):
#        if k != []:
#            for l in range(len(k)-1):
#                pic1_path = os.path.join(i, k[l])
#                pic2_path = os.path.join(i, k[l+1])
#                opt = opt_flow(pic1_path, pic2_path)
#                opt.pic()
#                df = opt.light_sql('Flash', '192.168.1.133', 'lls', 'Aa123456', 'lightning')
#                svpath = os.path.join(savepath, i.split('\\')[-2], i.split('\\')[-1])                    
#                if not os.path.exists(svpath):
#                    os.makedirs(svpath)
#                total_savepath = os.path.join(svpath, k[l][0:15])
#                if df.iloc[:,0].size > 100:
#                    print(df.iloc[:,0].size)
#                    opt.density(df, total_savepath)
#                    opt.draw_hsv(total_savepath)

      


    