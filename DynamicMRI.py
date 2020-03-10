import os
import SimpleITK as sitk
from DicomSeries import DicomSeries

class DynamicMRI:
    def __init__(self, path, annotated_only=False, verbose=False):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path)
        self.verbose = verbose
        if self.verbose:
            print(series_ids)
        #phases = {}
        self.phases = []

        for series_id in series_ids:
            try:
                series = DicomSeries(path, series_id, reader=reader, verbose=self.verbose)
                desc = series.get_desc()
                if True:
                #if ((('Ax ' in desc) or ('mDIXON' in desc)) and (('C-' in desc) or ('C+' in desc) or ('+C' in desc) or ('ARC' in desc)) and (not ('COR' in desc)) and (not ('Cor' in desc))) or ('t1_vibe_fs_tra' in desc):
                    if annotated_only:
                        if series.get_multi_anno() is not None:
                            if self.verbose:
                                print(desc)
                            self.phases.append(series)#.resize(256,256,series.image.GetDepth()))
                    else:
                        if self.verbose:
                            print(desc)
                        self.phases.append(series)
                    #self.phases.append(series.restore_original().resize(272,272,series.image.GetDepth()))
                    #self.phases.append(series.restore_original().resize(288,288,series.image.GetDepth()))
                    #self.phases.append(series.restore_original().resize(320,320,series.image.GetDepth()))
                #else:
                #    print(desc+' skipped')
            except:
                print('error')
                continue

if __name__ == '__main__':
    import sys
    dmri = DynamicMRI(sys.argv[1], verbose=True)
