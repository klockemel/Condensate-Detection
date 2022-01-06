# To detect condensates in 8 random images within the directory

# ls *.nd2 | sort -R | tail -n8 | while read file ; do python edge_detectCondensates.py $file ; done

ls *.nd2 | sort -R | tail -n8 | while read file ; do python edgeDetect_noSegmentation.py $file ; done