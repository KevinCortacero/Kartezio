from roifile import ImagejRoi


def read_ellipses_from_csv(dataframe, scale=1.0, ellipse_scale=1.0):
    ellipses = []
    for _, ellipse in dataframe.iterrows():
        ellipse_info = read_ellipse_from_row(ellipse, scale, ellipse_scale)
        ellipses.append(ellipse_info)
    return ellipses


def read_ellipse_from_row(row, scale=1.0, ellipse_scale=1.0):
    angle = 180 - row.Angle
    position = (row.X * scale, row.Y * scale)
    size = (
        row.Major * scale * ellipse_scale,
        row.Minor * scale * ellipse_scale,
    )
    ellipse_info = (position, size, angle)
    return ellipse_info


def read_polygons_from_roi(filename, scale=1.0):
    rois = ImagejRoi.fromfile(filename)
    if type(rois) == ImagejRoi:
        return [rois.coordinates()]
    polygons = [roi.coordinates() for roi in rois]
    return polygons
