
Extremely fast and accurate vectorized (SSE) gradient direction
(angle of rotation of a feature) finder for computer vision.

Operates on a 7x7 patch, i.e. a standard FAST feature. Larger
scale spaces can also call this function on the interpolated
image data for accurate angle-finding at larger scale.

Used in KORAL detector-descriptor pipeline.

Simply call like featureAngle(image, x, y, step)
where 'image' is a pointer to uint8_t (grayscale) image data,
'x' and 'y' are the horizontal and vertical coordinates of the center
of the feature, and 'step' is the row step (pitch) in bytes.

A float representing the angle *IN RADIANS*, between -PI and PI,
is returned.
