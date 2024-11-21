AWS=$1
DST_DIRECTORY=$2

$AWS s3 cp s3://opensynthetics-datasets/animated-gestures-poses-dataset-by-synthesis-ai $DST_DIRECTORY --recursive --no-sign-request