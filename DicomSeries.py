import os
import SimpleITK as sitk
import cv2
import numpy as np

class DicomSeries:
    def __init__(self, path, series_id, reader=None, verbose=False):
        if not reader:
            reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        self.path = path
        self.verbose = verbose
        self.file_names = reader.GetGDCMSeriesFileNames(path, series_id)
        self.annotation = None
        reader.SetFileNames(self.file_names)
        self.image = reader.Execute()
        self.orig_image = self.image
        #self.spacing = self.image.GetSpacing()
        #print(self.spacing)
        self.orig_tags = []
        self.tags = []
        for i in range(self.image.GetDepth()):
            self.orig_tags.append({})
            self.tags.append({})
            for key in reader.GetMetaDataKeys(i):
                self.orig_tags[-1][key] = reader.GetMetaData(i, key)
                if key == '0018|0050' or key == '0018|0088' or key == '0020|0032' or key == '0020|1041':
                    self.tags[-1][key] = self.orig_tags[-1][key]
        assert(len(self.file_names) == len(self.orig_tags))
        self.study_id = self.get_tag('0020|000d')
        self.series_id = self.get_tag('0020|000e')
        time = self.get_tag('0008|0031')
        #print(time)
        #assert(len(time)==6)
        self.strtime = time
        self.time = 60*60*int(time[0:2]) + 60*int(time[2:4]) + int(time[4:6])
        assert(self.series_id == series_id)

    def get_desc(self):
        return self.get_tag('0008|103e')
    
    def get_tag(self, tag_id):
        if tag_id[4] == ',':
            tag_id[4] = '|'
        if tag_id in self.orig_tags[0]:
            return self.orig_tags[0][tag_id]
        else:
            return None
    
    def get_anno(self):
        annofiles = []
        for name in self.file_names:
            path, filename = os.path.split(name)
            annofile = os.path.join(path, 'annotations', filename+'.png')
            if os.path.isfile(annofile):
                annofiles.append(annofile)
            #else:
            #    if len(annofiles) > 0:
            #        print('%s: %d annotations found, %s missing' % self.getdesc)
            #    return None
        if  len(self.file_names) != len(annofiles):
            if len(annofiles) > 0:
                print(self.path[-2:] + ': ' + self.get_desc() + '- Files: ' + str(len(self.file_names)) + ', Annos: ' + str(len(annofiles)))
            return None
        annos = []
        for annofile in annofiles:
            annos.append(cv2.imread(annofile)[:self.orig_image.GetHeight(),:self.orig_image.GetWidth(),0])

        return np.array(annos)
    
    def get_multi_anno(self):
        mixed_anno = self.get_anno()
        if mixed_anno is None:
            return None
        multianno = np.zeros((8,)+mixed_anno.shape, dtype=np.uint8)
        multianno[7,:,:,:] = mixed_anno // 128 > 0
        multianno[6,:,:,:] = (mixed_anno % 128) // 64 > 0
        multianno[5,:,:,:] = (mixed_anno % 64) // 32 > 0
        multianno[4,:,:,:] = (mixed_anno % 32) // 16 > 0
        multianno[3,:,:,:] = (mixed_anno % 16) // 8 > 0
        multianno[2,:,:,:] = (mixed_anno % 8) // 4 > 0
        multianno[1,:,:,:] = (mixed_anno % 4) // 2 > 0
        multianno[0,:,:,:] = (mixed_anno % 2) > 0
        self.annotation = multianno.astype(np.float32)
        if self.verbose:
            for i in range(8):
                print('Label '+str(i)+': '+str(self.annotation[i].max()))
        return self.annotation
    
    def numpy(self):
        return sitk.GetArrayFromImage(self.image)

    @staticmethod
    def get_z_position(position_string):
        last_bslash = position_string.rfind('\\')
        return float(position_string[last_bslash+1:])
    
    @staticmethod
    def modify_z_position(orig_string, new_z):
        last_bslash = orig_string.rfind('\\')
        return orig_string[:last_bslash+1] + str(new_z)
    
    def resize(self, width, height, depth, interpolation=sitk.sitkGaussian):
        #depth = self.image.GetDepth()
        orig_depth = self.image.GetDepth()
        depth_changed = not (depth == orig_depth)
        new_size = [width, height, depth]
        tmp_image = sitk.Image([width, height, depth], self.image.GetPixelIDValue())
        tmp_image.SetOrigin(self.image.GetOrigin())
        tmp_image.SetDirection(self.image.GetDirection())
        tmp_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, self.image.GetSize(), self.image.GetSpacing())])
        self.image = sitk.Resample(self.image, tmp_image, sitk.Transform(), interpolation)
        # Change position of every slice
        if depth_changed:
            for i in range(self.image.GetDepth()):
                new_z_position = float(self.tags[0]['0020|1041']) + i * self.image.GetSpacing()[2]
                if i >= len(self.tags):
                    self.tags.append({})
                # Slice thickness
                self.tags[i]['0018|0050'] = str(float(self.tags[0]['0018|0050']) * depth / orig_depth)
                # Spacing between slices
                self.tags[i]['0018|0088'] = str(self.image.GetSpacing()[2])
                # Image position
                self.tags[i]['0020|0032'] = DicomSeries.modify_z_position(self.tags[0]['0020|0032'], new_z_position)
                # Slice location
                self.tags[i]['0020|1041'] = str(new_z_position)
            # Remove unused entries
            while self.image.GetDepth() < len(self.tags):
                self.tags.pop(-1)
            assert(self.image.GetDepth() == len(self.tags))
        return self
    
    def restore_original(self):
        self.image = self.orig_image
        for i in range(self.image.GetDepth()):
            if i >= len(self.tags):
                    self.tags.append({})
            for key in self.tags[i]:
                self.tags[i][key] = self.orig_tags[i][key]
        while self.image.GetDepth() < len(self.tags):
                self.tags.pop(-1)
        assert(self.image.GetDepth() == len(self.tags))
        return self

    def crop(self, min_corner, max_corner):
        '''if len(min_corner) != 2:
            print('Error: x/y cropping only')
            return self
        elif isinstance(min_corner, tuple):
            min_corner = min_corner + (0,)
        elif isinstance(min_corner, list):
            min_corner = min_corner + [0]
        else:
            print('Error: tuple/list input only')
        
        if len(max_corner) != 2:
            print('Error: x/y cropping only')
            return self
        elif isinstance(max_corner, tuple):
            max_corner = max_corner + (0,)
        elif isinstance(max_corner, list):
            max_corner = max_corner + [0]
        else:
            print('Error: tuple/list input only')'''

        self.image = sitk.Crop(self.image, min_corner, max_corner)
        
        if max_corner[2] > 0:
            self.tags = self.tags[min_corner[2]:-max_corner[2]]
        else:
            self.tags = self.tags[min_corner[2]:]
        assert(self.image.GetDepth() == len(self.tags))
        
        return self

    def vcrop_from_position(self, start_pos, slices):
        image_depth = self.image.GetDepth()
        start_slice = 0
        for i in range(image_depth-1):
            if abs(start_pos - float(self.tags[i]['0020|1041'])) < abs(start_pos - float(self.tags[i+1]['0020|1041'])):
                start_slice = i
                break
        begin_crop = start_slice
        end_crop = image_depth - begin_crop - slices
        if end_crop < 0:
            print("Error: not enough to crop")
            return None
        self.image = sitk.Crop(self.image, (0,0,begin_crop), (0,0,end_crop))
        
        if end_crop > 0:
            self.tags = self.tags[begin_crop:-end_crop]
        else:
            self.tags = self.tags[begin_crop:]
        assert(self.image.GetDepth() == len(self.tags))
        
        return self

    
    def save(self, relative_path='out'):
        #print(self.file_names)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        for i in range(self.image.GetDepth()):
            image_slice = self.image[:,:,i]
            assert(i < len(self.tags))
            for key in self.orig_tags[i]:
                image_slice.SetMetaData(key, self.orig_tags[i][key])
                if key == '0018|0050' or key == '0018|0088' or key == '0020|0032' or key == '0020|1041':
                    image_slice.SetMetaData(key, self.tags[i][key])
            
            # Set 0018|0088 Spacing between slices using GetSpacing()[2]

            # Set 0020|0032 Image position
            #image_slice.SetMetaData("0020|0013", str(i+1)) # Instance Number

            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            if not os.path.isdir(os.path.join(os.path.split(self.file_names[i])[0], relative_path)):
                print('Making dir: %s' % (os.path.join(os.path.split(self.file_names[i])[0], relative_path)))
                os.makedirs(os.path.join(os.path.split(self.file_names[i])[0], relative_path))
            #else:
            #    print('Not making dir: %s' % (os.path.join(os.path.split(self.file_names[i])[0], relative_path)))
            writer.SetFileName(os.path.join(os.path.split(self.file_names[i])[0], relative_path, os.path.basename(self.file_names[i])))
            writer.Execute(image_slice)

if __name__ == '__main__':
    import sys
    series = DicomSeries(sys.argv[1], sys.argv[2], verbose=True)
    print(series.get_desc())
