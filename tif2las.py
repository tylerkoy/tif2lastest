import time
import concurrent.futures
from osgeo import gdal
import os
import shutil
import glob
import pandas as pd
import pdal
import multiprocessing
import sys
import argparse
import re
import cv2
import numpy as np
import pydicom as dicom
import subprocess

output12 = 0
output22 = 0
output32 = 0
img_names2 = []
z_step2 = 0
mode2 = 0
input2 = 0
cores2 = 0
v22 = 0
v12 = 0
s22 =0
s12 = 0
h22=0
h12=0
grey_scale_value2=0
process_name2=0
def arguments():
    print('starting up')
    parser = argparse.ArgumentParser()
    global output12,output22,color,mode2,output32,img_names2,z_step2,input2,cores2,v22,v12,s22,s12,h22,h12,grey_scale_value_lower2,grey_scale_value_upper2,process_name2,keeptif,dcm2tif,scale_x,scale_y,scale_z,NumFrame,classnumb
    parser.add_argument('-i','--input',required=True, help='(REQUIRED) input folder')
    parser.add_argument('-o','--output',required=True, help='(REQUIRED) output folder')
    parser.add_argument('-r','--rgb', action="store_true", default = False,help='rgb image')
    parser.add_argument('-c','--cores', default=4, type=int, help='how many cpu cores you want to dedicate')
    parser.add_argument('--gsupper', type=int, help='grayscale cutoff value, upper limit')
    parser.add_argument('--gslower', type=int, help='grayscale cutoff value, lower limit')
    parser.add_argument('--r1', default=0, type=int, help='bottom red value for RGB filtering')
    parser.add_argument('--r2', default=0, type=int, help='top red value for RGB filtering')
    parser.add_argument('--b1', default=0, type=int, help='bottom blue value for RGB filtering')
    parser.add_argument('--b2', default=0, type=int, help='top blue value for RGB filtering')
    parser.add_argument('--g1', default=0, type=int, help='bottom green for RGB filtering')
    parser.add_argument('--g2', default=0, type=int, help='top green for RGB filtering')
    parser.add_argument('--scale_x', default=0, help='scale of x from pdal writers.las options, default .01')
    parser.add_argument('--scale_y', default=0, help='scale of y from pdal writers.las options, default .01')
    parser.add_argument('--scale_z', default=0.01, type=float, help='scale of z from pdal writers.las options, default .01')
    parser.add_argument('--z_step', default=1, type=float, help='Z step from image to image')
    parser.add_argument('--keeptif', default=False, action="store_true", help='when using dicom2laz keep converted tif')
    parser.add_argument('--dcm2tif', default=False, action="store_true", help='dicom 2 tif')
    parser.add_argument('-n', '--classnumb', default=0, type=int, help='Total number of classifications to add')
    args = vars(parser.parse_args())
    color = args['rgb']
    classnumb = args['classnumb']
    classnumb = int(classnumb)
    grey_scale_value_lower2 = args['gslower']
    grey_scale_value_upper2 = args['gsupper']
    h12 = args['r1']
    h22 = args['r2']
    s12 = args['b1']
    s22 = args['b2']
    v12 = args['g1']
    v22 = args['g2']
    cores2 = args['cores']
    input2 = args['input']
    output12 = args['output']
    z_step2 = args['z_step']
    keeptif = args['keeptif']
    dcm2tif = args['dcm2tif']
    scale_x = args['scale_x']
    scale_y = args['scale_y']
    scale_z = args['scale_z']
    for images in glob.glob(input2 + '/*'):
        img_names2.append(images)
    output22 = output12 + '/'
    output32 = re.escape(output12)
    directory, images2 = os.path.split(images)
    fileName, fileExt = os.path.splitext(images2)
    print(scale_x)
    if str(fileExt) == '.dcm':
        mode2 = 'dcm'
    if mode2 == 'dcm':
        ds = dicom.dcmread((img_names2[0]), force=True)
        try:
            scale = ds.PixelSpacing
            scale_x = scale[0]
            scale_y = scale[1]
        except AttributeError:
            scale_x = input('No Pixel Spacing found, please enter preferred x scale (Default 1):')
            scale_y = input('No Pixel Spacing found, please enter preferred y scale (Default 1):')
            scale_x = float(scale_x)
            scale_y = float(scale_y)
            pass
    if dcm2tif != True:
        if not os.path.exists(output12):
            os.mkdir(output12)
        if not os.path.exists(output22 + 'xyz'):
            os.mkdir(output22 + 'xyz')
        if not os.path.exists(output22 + 'csv'):
            os.mkdir(output22 + 'csv')
        if not os.path.exists(output22 + 'las'):
            os.mkdir(output22 + 'las')
        if not os.path.exists(output22 + 'img'):
            os.mkdir(output22 + 'img')
    return output12,output22,output32,img_names2,z_step2,color,input2,cores2,v22,v12,s22,s12,h22,h12,grey_scale_value_lower2,grey_scale_value_upper2,process_name2,dcm2tif,keeptif,scale_x,scale_y,scale_z,mode2,classnumb


def pdalinsert(output4,filename,):
    json1 = """
             [
                 {{
                     "type":"readers.text",
                     "filename":"{file}/csv/{name}.csv"
                 }},
                 {{
                     "type":"writers.las",              
                     "filename":"{file}/las/{name}.laz"
                 }}
             ]
          """
    json2 = json1.format(file=output4, name=filename)
    return json2


def grayscale(input):
    img_name = input[0]
    output2 = input[1][0]
    z_step=input[1][1]
    mode=input[1][2]
    grey_scale_value_lower=input[1][9]
    img_names=input[1][10]
    scale_x = input[1][13]
    scale_y = input[1][14]
    scale_z = input[1][15]
    grey_scale_value_upper= input[1][16]
    output4 = output2.replace(os.sep,'/')
    z_count2 = img_names.index(img_name)
    z_count = (z_step * z_count2) - (z_step - 1)
    ds = gdal.Open(img_name)
    directory, img_name2 = os.path.split(img_name)
    fileName, fileExt = os.path.splitext(img_name2)
    print('Processing ' + fileName)
    out_ds = gdal.Translate(output2 + 'xyz/' + fileName + '.xyz', ds, format='XYZ')
    ds = None
    out_ds = None
    pd1 = pd.read_csv(output2 + 'xyz/' + fileName + '.xyz', sep=' ')
    pd1.columns = ['X', 'Y', 'intensity']
    pd1.insert(loc=2, column='Z', value=z_count)
    if grey_scale_value_lower is not None:
        pd1 = pd1[pd1.intensity >= grey_scale_value_lower]
    if grey_scale_value_upper is not None:
        pd1 = pd1[pd1.intensity <= grey_scale_value_upper]
    pd1.to_csv(output2 + 'csv/' + fileName + '.csv', sep=' ', index=False)
    json = pdalinsert(output4,fileName)
    pipeline = pdal.Pipeline(json)
    p = pipeline.execute()
    pdl = None
    os.remove(output2 + 'xyz/' + fileName + '.xyz')
    os.remove(output2 + 'csv/' + fileName + '.csv')
    print(f'{fileName} was processed')

def rgb(input):
    img_name = input[0]
    output2 = input[1][0]
    z_step=input[1][1]
    mode=input[1][2]
    img_names=input[1][10]
    r2=input[1][3]
    r1=input[1][4]
    b2=input[1][5]
    b1=input[1][6]
    g2=input[1][7]
    g1=input[1][8]
    scale_x = input[1][13]
    scale_y = input[1][14]
    scale_z = input[1][15]
    output4 = output2.replace(os.sep,'/')
    z_count2 = img_names.index(img_name)
    z_count = (z_step * z_count2) - (z_step - 1)
    directory, img_name2 = os.path.split(img_name)
    fileName, fileExt = os.path.splitext(img_name2)
    print('Processing ' + fileName)
    ds = gdal.Open(img_name)
    out_ds = gdal.Translate(output2 + 'xyz/' + fileName + 'r' + '.xyz', ds, format='XYZ', bandList=[1])
    out_dss = gdal.Translate(output2 + 'xyz/' + fileName + 'g' + '.xyz', ds, format='XYZ', bandList=[2])
    out_dsss = gdal.Translate(output2 + 'xyz/' + fileName + 'b' + '.xyz', ds, format='XYZ', bandList=[3])
    pd1 = pd.read_csv(output2 + 'xyz/' + fileName + 'r' + '.xyz', sep=' ')
    pd2 = pd.read_csv(output2 + 'xyz/' + fileName + 'g' + '.xyz', sep=' ')
    pd3 = pd.read_csv(output2 + 'xyz/' + fileName + 'b' + '.xyz', sep=' ')
    pd1.columns = ['X', 'Y', 'red']
    pd2.columns = ['X', 'Y', 'green']
    pd3.columns = ['X', 'Y', 'blue']
    pd1.insert(loc=2, column='Z', value=z_count)
    pd1['green'] = pd2['green']
    pd1['blue'] = pd3['blue']
    pd6 = pd1.loc[(pd1['red'] == 0) & (pd1['green'] == 0) & (pd1['blue'] == 0)]
    pd1 = pd1[~pd1.isin(pd6)].dropna()
    if ((g1 == 0) & (g2 == 0) & (b1 == 0) & (b2 == 0)):
        pd7 = pd1.loc[(pd1['red']>= r1) & (pd1['red']<= r2)]
    elif ((r1 == 0) & (r2 == 0) & (b1 == 0) & (b2 ==0)):
        pd7 = pd1.loc[(pd1['green']>= g1) & (pd1['green']<= g2)]
    elif ((r1 == 0) & (r2 == 0) & (g1 == 0) & (g2 == 0)):
        pd7 = pd1.loc[(pd1['blue']>= b1) & (pd1['blue']<= b2)]
    elif ((g1 == 0) & (g2 == 0)):
        pd7 = pd1.loc[(pd1['red']>= r1) & (pd1['red']<= r2) & (pd1['blue']>= b1) & (pd1['blue']<= b2)]
    elif ((b1 == 0) & (b2 == 0)):
        pd7 = pd1.loc[(pd1['red']>= r1) & (pd1['red']<= r2) & (pd1['green']>= g1) & (pd1['green']<= g2)]
    elif ((r1 == 0) & (r2 == 0)):
        pd7 = pd1.loc[(pd1['blue']>= b1) & (pd1['blue']<= b2) & (pd1['green']>= g1) & (pd1['green']<= g2)]
    else:
        pd7 = pd1.loc[(pd1['red']>= r1) & (pd1['red']<= r2) & (pd2['green']>= g1) & (pd2['green']<= g2) & (pd3['blue']>= b1)& (pd3['blue']<= b2)]
    pd1 = pd1[~pd1.isin(pd7)].dropna()
    pd1.to_csv(output2 + 'csv/' + fileName + '.csv', sep=' ', index=False)
    json = pdalinsert(output4,fileName)
    pipeline = pdal.Pipeline(json)
    p = pipeline.execute()
    img = None
    hsv = None
    mask = None
    inv_mask = None
    res = None
    pdl = None
    pd1 = None
    pd2 = None
    pd3 = None
    ds = None
    out_ds = None
    out_dss = None
    out_dsss = None
    os.remove(output2 + 'xyz/' + fileName + 'r' + '.xyz')
    os.remove(output2 + 'xyz/' + fileName + 'g' + '.xyz')
    os.remove(output2 + 'xyz/' + fileName + 'b' + '.xyz')
    os.remove(output2 + 'csv/' + fileName + '.csv')
    print(f'{fileName} was processed')

def dicom2laz(input):
    img_name = input[0]
    output2 = input[1][0]
    z_step=input[1][1]
    grey_scale_value_lower=input[1][9]
    img_names=input[1][10]
    keeptif = input[1][11]
    dcm2tif = input[1][12]
    scale_x = input[1][13]
    scale_y = input[1][14]
    scale_z = input[1][15]
    grey_scale_value_upper = input[1][16]
    directory, img_name2 = os.path.split(img_name)
    fileName, fileExt = os.path.splitext(img_name2)
    print('Processing ' + fileName)
    z_count2 = img_names.index(img_name)
    z_count = (z_step * z_count2) - (z_step - 1)
    ds = dicom.dcmread((img_name),force=True)
    arr = ds.pixel_array
    if scale_x == 0:
        scale = ds.PixelSpacing
        scale_x = scale[0]
        scale_y = scale[1]
    norm = np.linalg.norm(arr)
    norm_arr = arr / norm
    to_tiff = norm_arr * 65536
    try:
        NumFrame = ds.NumberOfFrames
    except AttributeError:
        print('Number of Frames not detected, defaulting to 1')
        NumFrame = 1
        pass
    if NumFrame == 1:
        m, n = to_tiff.shape
        R, C = np.mgrid[:m, :n]
        out = np.column_stack((C.ravel(), R.ravel(), to_tiff.ravel()))
        np.savetxt(output2 + fileName + '.xyz', out)
        ds = gdal.Open(output2 + fileName + '.xyz')
        gdal.Translate(output2 + fileName + '.tif', ds, format='Gtiff')
        ds = None
        arr = None
        norm = None
        norm_arr = None
        to_tiff = None
        m = None
        n = None
        R = None
        C = None
        out = None
        ds = None
        os.remove(output2 + fileName + '.xyz')
    else:
        for FrameNum in range(NumFrame):
            tempvar = to_tiff[FrameNum]
            m, n = tempvar.shape
            R, C = np.mgrid[:m, :n]
            out = np.column_stack((C.ravel(), R.ravel(), tempvar.ravel()))
            print('error here')
            FrameStr = str(FrameNum)
            np.savetxt(output2 + fileName + '_' + FrameStr + '.xyz', out)
            ds = gdal.Open(output2 + fileName + '_' + FrameStr + '.xyz')
            gdal.Translate(output2 + fileName + '_' + FrameStr + '.tif',ds,format='Gtiff')
            tempvar = None
            ds = None
            arr = None
            norm = None
            norm_arr = None
            m = None
            n = None
            R = None
            C = None
            out = None
            ds = None
            os.remove(output2 + fileName + '_' + FrameStr + '.xyz')
    if dcm2tif == True:
        print(f'{fileName} was processed')
        return
    img_name = output2 + fileName + '.tif'
    output4 = output2.replace(os.sep,'/')
    ds = gdal.Open(img_name)
    directory, img_name2 = os.path.split(img_name)
    fileName, fileExt = os.path.splitext(img_name2)
    out_ds = gdal.Translate(output2 + 'xyz/' + fileName + '.xyz', ds, format='XYZ')
    ds = None
    out_ds = None
    pd1 = pd.read_csv(output2 + 'xyz/' + fileName + '.xyz', sep=' ')
    pd1.columns = ['X', 'Y', 'intensity']
    pd1.insert(loc=2, column='Z', value=z_count)
    if grey_scale_value_lower is not None:
        pd1 = pd1[pd1.intensity >= grey_scale_value_lower]
    if grey_scale_value_upper is not None:
        pd1 = pd1[pd1.intensity <= grey_scale_value_upper]
    pd1.X = pd1.X*scale_x
    pd1.Y = pd1.Y*scale_y
    pd1.to_csv(output2 + 'csv/' + fileName + '.csv', sep=' ', index=False)
    json = pdalinsert(output4,fileName)
    pipeline = pdal.Pipeline(json)
    p = pipeline.execute()
    pdl = None
    os.remove(output2 + 'xyz/' + fileName + '.xyz')
    os.remove(output2 + 'csv/' + fileName + '.csv')
    if keeptif == True:
        print(f'{fileName} was processed')
        return
    os.remove(output2 + fileName + '.tif')
    print(f'{fileName} was processed ' + str(z_step) + ' ' + str(scale_x) + ' ' + str(scale_y))


def classification(output,classnumb):
    inp = output + '/las'
    classnumb = classnumb
    print(inp)
    jsonclassify = """
    	[
    		 {{
                         "type":"readers.las",
                         "filename":"{directory}/{name}"
                     }},
                     {{
                         "type":"filters.ferry",
                         "dimensions":"=>Classification"

                     }},
                     {{
                         "type":"filters.assign",
                         "assignment":"Classification[:]=0"

                     }},
                     {{
                         "type":"writers.las",              
                         "filename":"{directory}/{name}"
                     }}
    	]
              """

    json = """
    	[
    		 {{
                         "type":"readers.las",
                         "filename":"{directory}/{name}"
                     }},
                     {{
                         "type":"filters.assign",
                         "value":"Classification = {classification} WHERE {classtype} > {lowclass} && {classtype} < {uppclass}"

                     }},
                     {{
                         "type":"writers.las",              
                         "filename":"{directory}/{name}"
                     }}
    	]
              """

    jsonnullup = """
    	[
    		 {{
                         "type":"readers.las",
                         "filename":"{directory}/{name}"
                     }},
                     {{
                         "type":"filters.assign",
                         "value":"Classification = {classification}",
                         "where": "{classtype} > {lowclass}"

                     }},
                     {{
                         "type":"writers.las",              
                         "filename":"{directory}/{name}"
                     }}
    	]
              """

    jsonnulllow = """
    	[
    		 {{
                         "type":"readers.las",
                         "filename":"{directory}/{name}"
                     }},
                     {{
                         "type":"filters.assign",
                         "value":"Classification = {classification}",
                         "where": "{classtype} < {uppclass}"

                     }},
                     {{
                         "type":"writers.las",              
                         "filename":"{directory}/{name}"
                     }}
    	]
              """
    filename = os.path.basename(inp)
    classtype = []
    lowup = []
    classnumb = classnumb
    for numb in range(classnumb):
        string = 'Class ' + str(numb + 1) + ': Classify by Intesity? [y/n]'
        answer = input(string)
        if not answer or answer[0].lower() != 'y':
            classtype.append(input('Type Class Identifier:'))
            classtype[numb] = str(classtype[numb])
        else:
            classtype.append('Intensity')
        string = 'Class ' + str(numb + 1) + ': Lower ' + classtype[numb] + ' value (enter [n] if DNE):'
        answer1 = input(string)
        if answer1 == 'n':
            answer1 = None
        string = 'Class ' + str(numb + 1) + ': Upper ' + classtype[numb] + ' value (enter [n] if DNE):'
        answer2 = input(string)
        if answer2 == 'n':
            answer2 = None
        lowup.append([answer1, answer2])
        print(answer1)
        print(answer2)
    for filenames in glob.glob(inp + '\*.laz'):
        directory, name2 = os.path.split(filenames)
        json1 = jsonclassify.format(directory=directory, name=name2)
        json2 = json1.replace('\\', '/')
        pipeline = pdal.Pipeline(json2)
        p = pipeline.execute()
        for numb in range(classnumb):
            if lowup[numb][0] is None:
                json1 = jsonnulllow.format(directory=directory, name=name2, uppclass=int(lowup[numb][1]),
                                           classtype=classtype[numb], classification=(numb + 1))
                json2 = json1.replace('\\', '/')
                pipeline = pdal.Pipeline(json2)
                p = pipeline.execute()
            elif lowup[numb][1] is None:
                json1 = jsonnullup.format(directory=directory, name=name2, lowclass=int(lowup[numb][0]),
                                          classtype=classtype[numb], classification=(numb + 1))
                json2 = json1.replace('\\', '/')
                pipeline = pdal.Pipeline(json2)
                p = pipeline.execute()
            else:
                json1 = json.format(directory=directory, name=name2, lowclass=int(lowup[numb][0]),
                                    uppclass=int(lowup[numb][1]), classtype=classtype[numb], classification=(numb + 1))
                json2 = json1.replace('\\', '/')
                pipeline = pdal.Pipeline(json2)
                p = pipeline.execute()
        print(name2 + ' has been classified')


#command line options
#rgb or grayscale in pool.map
def main(img_names3,output23,z_step3,mode3,v23,v13,s23,s13,h23,h13,grey_scale_value_lower3,keeptif2,dcm2tif2,scale_x,scale_y,scale_z,grey_scale_value_upper3):
    iterable = img_names3
    args = [output23,z_step3,mode3,v23,v13,s23,s13,h23,h13,grey_scale_value_lower3,img_names3,keeptif2,dcm2tif2,scale_x,scale_y,scale_z,grey_scale_value_upper3]
    new_iterable=([x,args] for x in iterable)
    if color == True:
        process_name_2 = rgb
    else:
        process_name_2 = grayscale
    if mode2 == 'dcm':
        process_name_2 = dicom2laz
    if dcm2tif2 == True:
        process_name_2 = dicom2laz
    with multiprocessing.Pool(processes=cores2) as pool:
        result = pool.map(process_name_2,new_iterable)

if __name__=='__main__':
    multiprocessing.freeze_support()
    arguments()
    main(img_names2,output22,z_step2,mode2,v22,v12,s22,s12,h22,h12,grey_scale_value_lower2,keeptif,dcm2tif,scale_x,scale_y,scale_z,grey_scale_value_upper2)
    if not classnumb == 0:
        classification(output22,classnumb)


