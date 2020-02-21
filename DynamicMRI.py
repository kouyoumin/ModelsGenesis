import os
import SimpleITK as sitk
from DicomSeries import DicomSeries

class DynamicMRI:
    def __init__(self, path):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path)
        print(series_ids)
        #phases = {}
        self.phases = []

        for series_id in series_ids:
            try:
                series = DicomSeries(path, series_id, reader=reader)
                desc = series.get_desc()
                if ('Ax ' in desc) and (('C-' in desc) or ('C+' in desc)):
                    print(desc)
                    self.phases.append(series.resize(256,256,series.image.GetDepth()))
            except:
                continue

if __name__ == '__main__':
    import sys
    dmri = DynamicMRI(sys.argv[1])
